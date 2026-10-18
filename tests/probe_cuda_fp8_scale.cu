// Standalone E4M3 decoder validation. Does not select an inference kernel.
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__device__ __forceinline__ float original(unsigned code) {
    return __half2float(static_cast<__half>(
        __nv_cvt_fp8_to_halfraw(static_cast<unsigned char>(code), __NV_E4M3)));
}
__device__ __forceinline__ float integer_decode(unsigned code) {
    unsigned magnitude = code & 127;
    // Preserve the intrinsic's NaN payload/sign rather than guessing it.
    if (magnitude == 127) return original(code);
    float value = magnitude < 8 ? magnitude * (1.f / 512.f)
        : __uint_as_float((magnitude << 20) + 0x3c000000u);
    return __uint_as_float(__float_as_uint(value) | ((code & 128) << 24));
}
template<bool Integer>
__global__ void decode(float* result) {
    unsigned code = threadIdx.x;
    result[code] = Integer ? integer_decode(code) : original(code);
}
static void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}
int main() {
    float *a, *b;
    float reference[256], candidate[256];
    check(cudaMalloc(&a, sizeof(reference))); check(cudaMalloc(&b, sizeof(candidate)));
    decode<false><<<1, 256>>>(a); decode<true><<<1, 256>>>(b);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(reference, a, sizeof(reference), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(candidate, b, sizeof(candidate), cudaMemcpyDeviceToHost));
    for (unsigned code = 0; code < 256; ++code) {
        if (std::memcmp(reference + code, candidate + code, sizeof(float))) {
            std::fprintf(stderr, "code=%u mismatch\n", code); return 2;
        }
    }
    check(cudaFree(a)); check(cudaFree(b));
    std::puts("All 256 E4M3 codes bitwise match CUDA intrinsic (including signed zero and NaNs). No projection timing measured.");
}
