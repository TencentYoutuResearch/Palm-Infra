#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Platform detection for SIMD
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

// ---------------------------------------------------------------------------
// PROJECT_NAME — Tensor definition
// ---------------------------------------------------------------------------

enum class Precision : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

enum class MemoryType : uint8_t {
    NONE     = 0,   // unallocated
    OWNED    = 1,   // Tensor owns the memory (must delete[] data)
    POOLED   = 2,   // allocated from BufferPool (must return to pool)
    EXTERNAL = 3,   // external pointer (mmap, user-provided — never free)
};

// ---------------------------------------------------------------------------
// Tensor — fixed 4-D, row-major, stride in bytes
//
// shape[0] = innermost (columns / features)
// shape[1] = rows
// shape[2] = channels / slices
// shape[3] = batch
//
// Unused dimensions are set to 1.
//
// Views (reshape / permute / slice) are zero-copy: they share the parent's
// data pointer and only adjust shape + stride.
// ---------------------------------------------------------------------------

struct Tensor {
    Precision   prec     = Precision::FP32;
    MemoryType  mem_type = MemoryType::NONE;
    int64_t     shape[4] = {0, 1, 1, 1};
    size_t      stride[4] = {0, 0, 0, 0};  // stride in bytes
    void*       data     = nullptr;

    // -----------------------------------------------------------------------
    // factory
    // -----------------------------------------------------------------------

    static Tensor create(Precision prec, MemoryType mem, int64_t d0,
                         int64_t d1 = 1, int64_t d2 = 1, int64_t d3 = 1,
                         void* data = nullptr) {
        Tensor t;
        t.prec     = prec;
        t.mem_type = mem;
        t.shape[0] = d0;
        t.shape[1] = d1;
        t.shape[2] = d2;
        t.shape[3] = d3;
        t.compute_strides();
        t.data = data;
        return t;
    }

    // -----------------------------------------------------------------------
    // strides
    // -----------------------------------------------------------------------

    void compute_strides() {
        size_t es = element_size();
        stride[0] = es;
        stride[1] = stride[0] * shape[0];
        stride[2] = stride[1] * shape[1];
        stride[3] = stride[2] * shape[2];
    }

    // -----------------------------------------------------------------------
    // element / byte helpers
    // -----------------------------------------------------------------------

    size_t element_size() const {
        switch (prec) {
        case Precision::FP32: return 4;
        case Precision::FP16: return 2;
        case Precision::INT8: return 1;
        }
        return 0;
    }

    int64_t nelements() const {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }

    /// Total bytes occupied by this tensor (respects stride, handles
    /// non-contiguous layouts correctly).
    size_t nbytes() const {
        if (shape[0] <= 0) return 0;
        // stride[3] * shape[3] gives the total span in bytes
        return stride[3] * shape[3];
    }

    /// Check whether the tensor is densely packed in row-major order.
    bool is_contiguous() const {
        size_t expected = element_size();
        if (stride[0] != expected) return false;
        expected *= shape[0];
        for (int i = 1; i < 4; i++) {
            if (shape[i] != 1 && stride[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    /// True if dims [0..n) are contiguous (higher dims may be permuted).
    bool is_contiguous_n(int n) const {
        size_t expected = element_size();
        if (stride[0] != expected) return false;
        expected *= shape[0];
        for (int i = 1; i < 4; i++) {
            if (i > n) {
                if (shape[i] != 1 && stride[i] != expected) return false;
                expected *= shape[i];
            } else {
                expected = shape[i] * stride[i];
            }
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // data access
    // -----------------------------------------------------------------------

    template <typename T> T* ptr() {
        return static_cast<T*>(data);
    }
    template <typename T> const T* ptr() const {
        return static_cast<const T*>(data);
    }

    /// Access element at flat index (assumes contiguous layout).
    template <typename T> T& flat(int64_t i) {
        return ptr<T>()[i];
    }
    template <typename T> const T& flat(int64_t i) const {
        return ptr<T>()[i];
    }

    /// Row-major element access using byte strides.
    /// Usage:  t.at<T>(col, row)  for 2-D,  t.at<T>(col, row, chan, batch) for 4-D.
    template <typename T> T& at(int64_t i0) {
        return *reinterpret_cast<T*>(static_cast<char*>(data) + i0 * stride[0]);
    }
    template <typename T> T& at(int64_t i0, int64_t i1) {
        return *reinterpret_cast<T*>(static_cast<char*>(data) + i0 * stride[0] + i1 * stride[1]);
    }
    template <typename T> T& at(int64_t i0, int64_t i1, int64_t i2) {
        return *reinterpret_cast<T*>(static_cast<char*>(data) + i0 * stride[0] + i1 * stride[1] + i2 * stride[2]);
    }
    template <typename T> T& at(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        return *reinterpret_cast<T*>(static_cast<char*>(data) + i0 * stride[0] + i1 * stride[1] + i2 * stride[2] + i3 * stride[3]);
    }

    // const variants
    template <typename T> const T& at(int64_t i0) const {
        return *reinterpret_cast<const T*>(static_cast<const char*>(data) + i0 * stride[0]);
    }
    template <typename T> const T& at(int64_t i0, int64_t i1) const {
        return *reinterpret_cast<const T*>(static_cast<const char*>(data) + i0 * stride[0] + i1 * stride[1]);
    }
    template <typename T> const T& at(int64_t i0, int64_t i1, int64_t i2) const {
        return *reinterpret_cast<const T*>(static_cast<const char*>(data) + i0 * stride[0] + i1 * stride[1] + i2 * stride[2]);
    }
    template <typename T> const T& at(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const {
        return *reinterpret_cast<const T*>(static_cast<const char*>(data) + i0 * stride[0] + i1 * stride[1] + i2 * stride[2] + i3 * stride[3]);
    }

    /// Row pointer (for 2-D: data + row * stride[1]).
    template <typename T> T* row(int64_t r) {
        return reinterpret_cast<T*>(static_cast<char*>(data) + r * stride[1]);
    }
    template <typename T> const T* row(int64_t r) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data) + r * stride[1]);
    }

    /// Channel pointer (for 3-D+).
    template <typename T> T* channel(int64_t c) {
        return reinterpret_cast<T*>(static_cast<char*>(data) + c * stride[2]);
    }
    template <typename T> const T* channel(int64_t c) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data) + c * stride[2]);
    }

    // -----------------------------------------------------------------------
    // views — zero-copy, share data pointer
    // -----------------------------------------------------------------------

    /// 1-D view starting at byte offset, with ne0 elements.
    Tensor view_1d(int64_t ne0, size_t offset = 0) const {
        Tensor v = *this;
        v.shape[0] = ne0;
        v.shape[1] = 1;
        v.shape[2] = 1;
        v.shape[3] = 1;
        v.stride[0] = stride[0];
        v.stride[1] = v.stride[0] * ne0;
        v.stride[2] = v.stride[1];
        v.stride[3] = v.stride[2];
        v.data = static_cast<char*>(data) + offset;
        return v;
    }

    /// 2-D view starting at byte offset.
    Tensor view_2d(int64_t ne0, int64_t ne1, size_t offset = 0) const {
        Tensor v = *this;
        v.shape[0] = ne0;
        v.shape[1] = ne1;
        v.shape[2] = 1;
        v.shape[3] = 1;
        v.stride[0] = stride[0];
        v.stride[1] = stride[1];   // keep parent row stride
        v.stride[2] = v.stride[1] * ne1;
        v.stride[3] = v.stride[2];
        v.data = static_cast<char*>(data) + offset;
        return v;
    }

    /// Reshape — zero-copy.  Total elements must match.
    Tensor reshape(int64_t s0, int64_t s1 = 1, int64_t s2 = 1, int64_t s3 = 1) const {
        assert(nelements() == s0 * s1 * s2 * s3);
        Tensor v = *this;
        v.shape[0] = s0;
        v.shape[1] = s1;
        v.shape[2] = s2;
        v.shape[3] = s3;
        // strides are inherited; caller must ensure contiguity or use
        // appropriately.  A reshape on a non-contiguous tensor is only
        // valid when the new shape matches the stride pattern.
        return v;
    }

    /// Permute axes — zero-copy, swaps shape + stride.
    Tensor permute(int a0, int a1, int a2, int a3) const {
        Tensor v = *this;
        int64_t  new_shape[4];
        size_t   new_stride[4];
        new_shape[a0]  = shape[0];   new_stride[a0]  = stride[0];
        new_shape[a1]  = shape[1];   new_stride[a1]  = stride[1];
        new_shape[a2]  = shape[2];   new_stride[a2]  = stride[2];
        new_shape[a3]  = shape[3];   new_stride[a3]  = stride[3];
        for (int i = 0; i < 4; i++) { v.shape[i] = new_shape[i]; v.stride[i] = new_stride[i]; }
        return v;
    }

    // -----------------------------------------------------------------------
    // comparison
    // -----------------------------------------------------------------------

    bool operator==(const Tensor& other) const {
        return prec == other.prec && mem_type == other.mem_type &&
               shape[0] == other.shape[0] && shape[1] == other.shape[1] &&
               shape[2] == other.shape[2] && shape[3] == other.shape[3] &&
               stride[0] == other.stride[0] && stride[1] == other.stride[1] &&
               stride[2] == other.stride[2] && stride[3] == other.stride[3] &&
               data == other.data;
    }
    bool operator!=(const Tensor& other) const { return !(*this == other); }
};

// ---------------------------------------------------------------------------
// utility
// ---------------------------------------------------------------------------

inline size_t precision_size(Precision p) {
    switch (p) {
    case Precision::FP32: return 4;
    case Precision::FP16: return 2;
    case Precision::INT8: return 1;
    }
    return 0;
}
