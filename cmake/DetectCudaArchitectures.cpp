#include <cuda_runtime_api.h>
#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) return 1;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return 1;
        if (device) std::printf(";");
        std::printf("%d%d", properties.major, properties.minor);
    }
    return 0;
}
