#pragma once

// Shared helpers for the standalone CUDA tests: fatal checks, guard-banded
// device buffers and bit-exact comparisons. A failed check prints the message
// and exits with status 1, which CTest reports as a failure.

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace mollm_test {

// Exit status that CTest treats as "skipped" (SKIP_RETURN_CODE 77).
inline constexpr int skip_status = 77;

[[noreturn]] inline void fail(const char* message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

inline void require(bool ok, const char* message) {
    if (!ok) fail(message);
}

inline void check(cudaError_t status, const char* what = "CUDA call") {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

inline bool cuda_device_available() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return devices > 0;
}

// Device allocation with `guard_bytes` of `guard_pattern` on each side of the
// payload. check_guards() verifies that no kernel wrote outside the payload.
class GuardedBuffer {
public:
    static constexpr size_t guard_bytes = 256;
    static constexpr unsigned char guard_pattern = 0x5a;

    explicit GuardedBuffer(size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&allocation_, bytes + 2 * guard_bytes), "cudaMalloc");
        check(cudaMemset(allocation_, guard_pattern, bytes + 2 * guard_bytes), "cudaMemset");
        check(cudaDeviceSynchronize(), "guard initialization");
    }
    ~GuardedBuffer() { cudaFree(allocation_); }
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;

    void* data() { return allocation_ + guard_bytes; }
    const void* data() const { return allocation_ + guard_bytes; }
    template<class T> T* as() { return static_cast<T*>(data()); }
    template<class T> const T* as() const { return static_cast<const T*>(data()); }
    size_t bytes() const { return bytes_; }

    // Copies exactly bytes() from host memory. Synchronizes the device so the
    // data is visible to work on non-blocking streams as well.
    void upload(const void* source) {
        check(cudaMemcpy(data(), source, bytes_, cudaMemcpyHostToDevice), "upload");
        check(cudaDeviceSynchronize(), "upload");
    }
    // Fills the payload (not the guards) with `value` bytes.
    void fill(unsigned char value) {
        check(cudaMemset(data(), value, bytes_), "fill");
        check(cudaDeviceSynchronize(), "fill");
    }
    template<class T> std::vector<T> download() const {
        require(bytes_ % sizeof(T) == 0, "download size is not a multiple of the element size");
        std::vector<T> result(bytes_ / sizeof(T));
        check(cudaMemcpy(result.data(), data(), bytes_, cudaMemcpyDeviceToHost), "download");
        return result;
    }
    void check_guards() const {
        unsigned char before[guard_bytes], after[guard_bytes];
        check(cudaMemcpy(before, allocation_, guard_bytes, cudaMemcpyDeviceToHost), "guard read");
        check(cudaMemcpy(after, allocation_ + guard_bytes + bytes_, guard_bytes,
                         cudaMemcpyDeviceToHost), "guard read");
        for (size_t i = 0; i < guard_bytes; ++i)
            require(before[i] == guard_pattern && after[i] == guard_pattern,
                    "write outside the buffer payload");
    }

private:
    unsigned char* allocation_ = nullptr;
    size_t bytes_;
};

template<class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}

inline float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint32_t to_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// FP32 values that exactness tests feed through kernels: signed zeros,
// subnormals, extremes, infinities and a quiet NaN.
inline std::vector<float> special_fp32_values() {
    return {0.0f, -0.0f, std::numeric_limits<float>::denorm_min(),
            -std::numeric_limits<float>::denorm_min(), FLT_MIN, -FLT_MIN,
            FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()};
}

}  // namespace mollm_test
