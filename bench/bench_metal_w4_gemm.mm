// Microbenchmark for prefill-time dense Metal W4A16 GEMM.
//
// Rotate through independent layer-sized weights so the result reflects
// production streaming rather than a cache-hot matrix. Weights use the
// decoded Metal layout: [offset-binary W4 nibbles][FP32 G128 scales].

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernels/metal/metal_common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

namespace {

struct Case {
    const char* name;
    int m;
    int n;
    int k;
};

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "bench_metal_w4_gemm: %s\n", message);
    std::exit(1);
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes,
                          const char* label) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
    if (!buffer) fail("Metal buffer allocation failed");
    buffer.label = [NSString stringWithUTF8String:label];
    return buffer;
}

void initialize_weights(id<MTLBuffer> weights, const Case& c,
                        int layers) {
    const int groups = c.k / 128;
    const size_t qbytes = static_cast<size_t>(c.n) * c.k / 2;
    const size_t scale_count =
        static_cast<size_t>(c.n) * groups;
    const size_t layer_bytes =
        qbytes + scale_count * sizeof(float);
    auto* base = static_cast<uint8_t*>(weights.contents);
    for (int layer = 0; layer < layers; ++layer) {
        auto* layer_base =
            base + static_cast<size_t>(layer) * layer_bytes;
        for (size_t i = 0; i < qbytes; ++i)
            layer_base[i] = static_cast<uint8_t>(
                (i * 17 + layer * 13) & 0xff);
        auto* scales =
            reinterpret_cast<float*>(layer_base + qbytes);
        for (size_t i = 0; i < scale_count; ++i)
            scales[i] =
                0.0025f * static_cast<float>(1 + i % 3);
    }
}

void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> pipeline,
              const Case& c, int tile_m, int layers,
              int warmup, int iterations) {
    const int groups = c.k / 128;
    const size_t qbytes = static_cast<size_t>(c.n) * c.k / 2;
    const size_t scale_bytes =
        static_cast<size_t>(c.n) * groups * sizeof(float);
    const size_t layer_bytes = qbytes + scale_bytes;
    id<MTLBuffer> activation =
        make_buffer(
            device,
            static_cast<size_t>(c.m) * c.k * sizeof(float),
            "dense W4 GEMM activation");
    id<MTLBuffer> weights =
        make_buffer(
            device, layer_bytes * layers,
            "dense W4 GEMM weights");
    id<MTLBuffer> output =
        make_buffer(
            device,
            static_cast<size_t>(c.m) * c.n * sizeof(float),
            "dense W4 GEMM output");

    auto* activation_data =
        static_cast<float*>(activation.contents);
    for (size_t i = 0;
         i < static_cast<size_t>(c.m) * c.k; ++i)
        activation_data[i] =
            0.001f * static_cast<float>(
                static_cast<int>((i * 29) % 101) - 50);
    initialize_weights(weights, c, layers);

    MatmulW8Params params{};
    params.M = c.m;
    params.N = c.n;
    params.K = c.k;
    params.a_row_stride = c.k;
    params.c_row_stride = c.n;
    params.group_size = 128;
    params.groups_per_row = groups;

    auto encode =
        ^(id<MTLCommandBuffer> command_buffer, int repeats) {
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:activation offset:0 atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:2];
            [encoder setBytes:&params
                       length:sizeof(params)
                      atIndex:3];
            [encoder setThreadgroupMemoryLength:
                         64 * 128 * sizeof(uint16_t)
                                            atIndex:0];
            for (int repeat = 0; repeat < repeats; ++repeat) {
                const NSUInteger layer =
                    static_cast<NSUInteger>(repeat % layers);
                const NSUInteger layer_offset =
                    layer * layer_bytes;
                [encoder setBuffer:weights
                            offset:layer_offset
                           atIndex:1];
                [encoder setBuffer:weights
                            offset:layer_offset + qbytes
                           atIndex:4];
                [encoder dispatchThreadgroups:
                             MTLSizeMake(
                                 (c.m + tile_m - 1) / tile_m,
                                 (c.n + 63) / 64, 1)
                    threadsPerThreadgroup:
                             MTLSizeMake(128, 1, 1)];
            }
            [encoder endEncoding];
        };

    if (warmup > 0) {
        id<MTLCommandBuffer> command_buffer =
            [queue commandBuffer];
        encode(command_buffer, std::max(warmup, layers));
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status !=
            MTLCommandBufferStatusCompleted)
            fail("warmup failed");
    }

    id<MTLCommandBuffer> command_buffer =
        [queue commandBuffer];
    encode(command_buffer, iterations);
    const auto wall_start =
        std::chrono::steady_clock::now();
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const auto wall_end =
        std::chrono::steady_clock::now();
    if (command_buffer.status !=
        MTLCommandBufferStatusCompleted)
        fail("benchmark failed");

    const double gpu_us =
        (command_buffer.GPUEndTime -
         command_buffer.GPUStartTime) *
        1.0e6 / iterations;
    const double wall_us =
        std::chrono::duration<double, std::micro>(
            wall_end - wall_start).count() /
        iterations;
    const double bandwidth =
        static_cast<double>(layer_bytes) /
        (gpu_us * 1.0e-6) / 1.0e9;
    const volatile float checksum =
        static_cast<const float*>(output.contents)[
            static_cast<size_t>(c.m) * c.n - 1];

    std::printf(
        "%-9s M=%-3d N=%-5d K=%-5d TM=%-3d "
        "weights=%5.1f MiB gpu=%7.1f us wall=%7.1f us "
        "effective=%6.1f GB/s checksum=%g\n",
        c.name, c.m, c.n, c.k, tile_m,
        static_cast<double>(layer_bytes) /
            (1024.0 * 1024.0),
        gpu_us, wall_us, bandwidth,
        static_cast<double>(checksum));
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        const int iterations =
            argc > 1 ? std::max(1, std::atoi(argv[1])) : 144;
        const int warmup =
            argc > 2 ? std::max(0, std::atoi(argv[2])) : 48;
        const int tile_m =
            argc > 3 ? std::atoi(argv[3]) : 64;
        const int layers =
            argc > 4 ? std::max(1, std::atoi(argv[4])) : 48;
        const bool specialize_g128 =
            argc <= 5 || std::atoi(argv[5]) != 0;
        if (tile_m != 64 && tile_m != 128)
            fail("TM must be 64 or 128");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail("no Metal device");
        id<MTLCommandQueue> queue =
            [device newCommandQueue];
        if (!queue) fail("could not create command queue");

        NSError* error = nil;
        NSString* library_path =
            [NSString stringWithUTF8String:
                MOLLM_METALLIB_PATH];
        id<MTLLibrary> library =
            [device
                newLibraryWithURL:
                    [NSURL fileURLWithPath:library_path]
                              error:&error];
        if (!library) fail("could not load metallib");
        const char* function_name =
            tile_m == 128
                ? "gemm_tensor_w4_f32a_i4b_f32c"
                : "gemm_tensor_w4_f32a_i4b_f32c_m64";
        NSString* metal_function_name =
            [NSString stringWithUTF8String:function_name];
        MTLFunctionConstantValues* constants =
            [[MTLFunctionConstantValues alloc] init];
        bool enabled = specialize_g128;
        [constants setConstantValue:&enabled
                              type:MTLDataTypeBool
                           atIndex:10];
        id<MTLFunction> function =
            [library
                newFunctionWithName:metal_function_name
                     constantValues:constants
                              error:&error];
        if (!function) fail("could not load W4 GEMM function");
        id<MTLComputePipelineState> pipeline =
            [device
                newComputePipelineStateWithFunction:function
                                               error:&error];
        if (!pipeline) fail("could not create W4 GEMM pipeline");

        std::printf(
            "device=%s iterations=%d warmup=%d layers=%d "
            "specialize_g128=%s\n",
            device.name.UTF8String, iterations, warmup, layers,
            specialize_g128 ? "true" : "false");
        const Case cases[] = {
            {"qkv-256", 256, 6144, 2048},
            {"qkv-151", 151, 6144, 2048},
            {"qkv-93", 93, 6144, 2048},
            {"out-256", 256, 2048, 4096},
            {"out-151", 151, 2048, 4096},
            {"out-93", 93, 2048, 4096},
        };
        for (const Case& c : cases)
            run_case(
                device, queue, pipeline, c, tile_m,
                layers, warmup, iterations);
    }
    return 0;
}
