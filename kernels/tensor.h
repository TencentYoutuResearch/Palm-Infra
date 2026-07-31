#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernels/cpu_platform.h"
#include "kernels/prepared_weight.h"

// ---------------------------------------------------------------------------
// mollm — Tensor definition
// ---------------------------------------------------------------------------

enum class Precision : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    // DeepSeek-V4 dense matrices: E4M3 values with an explicitly described
    // block-scale layout in the weight metadata.
    FP8_E4M3 = 4,
    // OCP MXFP4: packed E2M1 values, one E8M0 scale per 32 K elements.
    MXFP4 = 5,
    // Integer graph metadata such as token-to-expert routing tables.
    INT32 = 6,
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
    // GPU backend (Metal) device storage. Inert on the CPU path (stay null/0),
    // so all existing CPU logic and the default CPU build are unaffected.
    //   device_data   = opaque id<MTLBuffer> handle backing this tensor.
    //   device_offset = byte offset into that buffer's contents (views cannot
    //     pointer-offset an MTLBuffer the way a char* can, so the handle stays
    //     the same and the offset is carried separately).
    void*       device_data   = nullptr;
    size_t      device_offset = 0;
    uint32_t    owner_id = 0;  // debug owner for pooled storage; 0 = unknown/non-pooled
    uint64_t    storage_id = 0; // debug allocation identity; copied by borrowed views
    const float* scales = nullptr; // quant scales for INT8/INT4 weights; borrowed from weight file
    // Raw E8M0 block scales for FP8_E4M3 and MXFP4. E8M0 is a power-of-two
    // scale encoding, so preserving it avoids expanding the 284B checkpoint.
    const uint8_t* e8m0_scales = nullptr;
    // Optional load-time Q8-dot sidecar for native FP8 weights. It keeps the
    // package bytes in FP8 while avoiding per-token FP8 decode on CPUs without
    // native FP8 arithmetic.
    const float* fp8_q8_scales = nullptr;
    // DeepSeek-V4 wo_a is defined as FP8 dequantized to BF16 before its
    // grouped einsum. BF16 values are exactly representable in FP16, so this
    // optional interleaved FP16 layout preserves those values while enabling
    // the existing NEON dense kernel.
    const void* fp8_bf16_fp16_data = nullptr;
    uint32_t    group_size = 0;    // K-dim quant group size; K means per-channel
    uint32_t    num_groups = 0;    // total groups = N * groups_per_row
    uint32_t    groups_per_row = 0;
    bool        is_interleaved = false; // weight data is packed as [N/8, K, 8]
    const void* rowmajor_data = nullptr; // original linear weight, retained beside CPU packs
    bool        is_q4_repacked = false; // INT4 data itself is [N/8, K/32, 8, 16B]
    bool        is_q4_g32_packed = false; // INT4 data itself is [N/8, K/32] G32 blocks
    bool        is_q4_g128_packed = false; // INT4 data itself is [N/8, K/128] G128 blocks
    bool        is_fp8_block128 = false; // scales are [ceil(N/128),ceil(K/128)]
    const void* q8_repack_data = nullptr; // optional [N/8, K/32, 8, 32] INT8 dot layout
    const void* q4_repack_data = nullptr; // optional [N/8, K/32, 8, 16B] INT4 dot layout
    const void* q4_g32_data = nullptr; // optional [N/8, K/32] G32 packed INT4+scales
    const void* q4_g128_data = nullptr; // optional [N/8, K/128] G128 packed INT4+scales
    // Non-owning reference to backend-specific load-time layouts. Row views
    // retain the same prepared weight and advance this logical output-row
    // offset instead of manufacturing backend-specific sidecar pointers.
    const PreparedWeight* prepared_weight = nullptr;
    size_t prepared_weight_row_offset = 0;
    const void* sparse_data = nullptr; // optional [N/8,K,8] sparse-A GEMV layout
    // Optional CPU MoE SSD source. Aggregate expert tensors use this instead
    // of data when expert weights are paged in on demand.
    const void* moe_ssd_source = nullptr;

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
        case Precision::INT4: return 1; // packed storage byte; logical element is a nibble
        case Precision::FP8_E4M3: return 1;
        case Precision::MXFP4: return 1; // packed storage byte; logical element is a nibble
        case Precision::INT32: return 4;
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

    /// Byte span reachable from this tensor view's first element through its
    /// last logical element. Unlike nbytes(), this does not include unused
    /// row tails before the view's starting offset.
    size_t view_span_bytes() const {
        if (nelements() <= 0) return 0;
        size_t span = element_size();
        for (int dimension = 0; dimension < 4; ++dimension)
            span += static_cast<size_t>(shape[dimension] - 1) *
                stride[dimension];
        return span;
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

    /// Whether two tensors refer to the same underlying allocation.
    ///
    /// Pooled tensors carry a stable storage identity, which remains valid for
    /// views whose data pointers have been offset. External tensors do not
    /// have that identity, so pointer equality is the best available fallback.
    bool shares_storage_with(const Tensor& other) const {
        if (storage_id != 0 && other.storage_id != 0) {
            return owner_id == other.owner_id && storage_id == other.storage_id;
        }
        return data == other.data;
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
        v.device_offset = device_offset + offset;  // inert on CPU (offset 0-based, data drives)
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
        v.device_offset = device_offset + offset;  // inert on CPU
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
    case Precision::INT4: return 1; // packed storage byte
    case Precision::FP8_E4M3: return 1;
    case Precision::MXFP4: return 1; // packed storage byte
    case Precision::INT32: return 4;
    }
    return 0;
}
