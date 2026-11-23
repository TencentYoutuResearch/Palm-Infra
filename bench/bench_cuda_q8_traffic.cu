// Read-cost diagnostic, not a faster or numerically equivalent inference path.
#include "kernels/cuda/matmul.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

static void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}
template<bool Wide = false>
__global__ void read_rows(const uint32_t* weights, uint32_t* output, int rows, int words) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x % 32;
    uint32_t sum = 0;
    if constexpr (Wide) {
        const auto* vectors = reinterpret_cast<const uint4*>(weights + size_t(row) * words);
        for (int word = lane; word < words / 4; word += 32) {
            const uint4 value = vectors[word];
            sum += value.x + value.y + value.z + value.w;
        }
    } else {
        for (int word = lane; word < words; word += 32)
            sum += weights[size_t(row) * words + word];
    }
    for (int distance = 16; distance; distance >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, distance);
    if (!lane) output[row] = sum;
}
template<class Launch>
static float measure(Launch launch, cudaStream_t stream) {
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 64; ++i) launch(stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    check(cudaEventRecord(start, stream));
    for (int i = 0; i < 4; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
    float ms; check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    return ms * 1000 / 256;
}
int main() {
    constexpr int rows = 248320, inner = 2560, words = inner / 4;
    const size_t count = size_t(rows) * words;
    std::vector<uint32_t> host(count), expected(rows), actual(rows);
    std::vector<float> input(inner), scales(rows);
    for (int row = 0; row < rows; ++row) {
        for (int word = 0; word < words; ++word) {
            size_t index = size_t(row) * words + word;
            host[index] = uint32_t(index) * 2654435761u + 12345u;
            expected[row] += host[index];
        }
        scales[row] = (1 + row % 17) / 37.f;
    }
    for (int i = 0; i < inner; ++i) input[i] = std::sin(float(i) * .173f);
    uint32_t *weights, *checksums;
    float *x, *s, *y;
    check(cudaMalloc(&weights, count * 4)); check(cudaMalloc(&checksums, rows * 4));
    check(cudaMalloc(&x, inner * 4)); check(cudaMalloc(&s, rows * 4)); check(cudaMalloc(&y, rows * 4));
    check(cudaMemcpy(weights, host.data(), count * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(x, input.data(), inner * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(s, scales.data(), rows * 4, cudaMemcpyHostToDevice));
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (std::getenv("MOLLM_Q8_BATCH_PROBE")) {
        const int lda=inner+4, ldc=rows+9;
        float *batch_x,*batch_y;
        check(cudaMalloc(&batch_x,size_t(8)*lda*sizeof(float)));
        check(cudaMalloc(&batch_y,size_t(8)*ldc*sizeof(float)));
        std::vector<float> inputs(size_t(8)*lda,999.f), baseline(size_t(8)*ldc), result(baseline.size());
        for(int m:{2,3,4,5,7,8}) {
            auto serial=[&](cudaStream_t selected) {
                for(int t=0;t<m;++t)check(mollm_cuda::launch_q8_tuned_gemv(
                    batch_x+size_t(t)*lda,reinterpret_cast<const int8_t*>(weights),s,
                    batch_y+size_t(t)*ldc,rows,inner,8,false,selected));
            };
            auto batch=[&](cudaStream_t selected) {
                check(mollm_cuda::launch_q8_small_batch_exact(batch_x,
                    reinterpret_cast<const int8_t*>(weights),s,batch_y,m,rows,inner,lda,ldc,selected));
            };
            for(int replay=0;replay<2;++replay) {
                for(int t=0;t<m;++t)for(int c=0;c<inner;++c)
                    inputs[size_t(t)*lda+c]=std::sin(float(c+t*71)*.173f)*(replay?-1.f:1.f);
                check(cudaMemcpyAsync(batch_x,inputs.data(),inputs.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
                std::fill(baseline.begin(),baseline.end(),-123.f);
                check(cudaMemcpyAsync(batch_y,baseline.data(),baseline.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
                serial(stream);check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(baseline.data(),batch_y,baseline.size()*sizeof(float),cudaMemcpyDeviceToHost));
                std::fill(result.begin(),result.end(),-123.f);
                check(cudaMemcpyAsync(batch_y,result.data(),result.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
                // measure executes captured graphs and leaves the final replay output.
                float serial_us,batch_us;
                if(replay) {batch_us=measure(batch,stream);serial_us=measure(serial,stream);batch(stream);}
                else {serial_us=measure(serial,stream);batch_us=measure(batch,stream);}
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(result.data(),batch_y,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
                for(size_t i=0;i<result.size();++i)if(!std::isfinite(result[i])||
                    std::memcmp(&baseline[i],&result[i],sizeof(float))) {
                    std::fprintf(stderr,"Q8 batch mismatch M=%d replay=%d i=%zu %g != %g\n",m,replay,i,result[i],baseline[i]);return 4;
                }
                std::printf("M=%d replay=%d serial_us=%.3f batch_us=%.3f speedup=%.3f bit_exact=1 guards=1\n",
                    m,replay,serial_us,batch_us,serial_us/batch_us);std::fflush(stdout);
            }
        }
        check(cudaFree(batch_x));check(cudaFree(batch_y));check(cudaStreamDestroy(stream));
        check(cudaFree(weights));check(cudaFree(checksums));check(cudaFree(x));check(cudaFree(s));check(cudaFree(y));
        return 0;
    }
    auto read = [&](cudaStream_t selected) {
        read_rows<><<<(rows + 7) / 8, 256, 0, selected>>>(weights, checksums, rows, words);
        check(cudaGetLastError());
    };
    auto wide = [&](cudaStream_t selected) {
        read_rows<true><<<(rows + 7) / 8, 256, 0, selected>>>(weights, checksums, rows, words);
        check(cudaGetLastError());
    };
    auto head = [&](cudaStream_t selected) {
        check(mollm_cuda::launch_q8_tuned_gemv(x, reinterpret_cast<const int8_t*>(weights), s, y,
                                               rows, inner, 8, false, selected));
    };
    for (int repeat = 0; repeat < 3; ++repeat) {
        float read_us, head_us;
        if (repeat % 2) { read_us = measure(read, stream); head_us = measure(head, stream); }
        else { head_us = measure(head, stream); read_us = measure(read, stream); }
        check(cudaMemcpy(actual.data(), checksums, rows * 4, cudaMemcpyDeviceToHost));
        if (actual != expected) { std::fprintf(stderr, "read checksum mismatch\n"); return 2; }
        std::printf("repeat=%d read_us=%.3f head_us=%.3f logical_weight_GBps=%.3f checksum_exact=1\n",
                    repeat, read_us, head_us, double(count * 4) / read_us / 1000);
        const float wide_us = measure(wide, stream);
        check(cudaMemcpy(actual.data(), checksums, rows * 4, cudaMemcpyDeviceToHost));
        if (actual != expected) { std::fprintf(stderr, "wide read checksum mismatch\n"); return 3; }
        std::printf("repeat=%d wide_read_us=%.3f logical_weight_GBps=%.3f checksum_exact=1\n",
                    repeat, wide_us, double(count * 4) / wide_us / 1000);
    }
    check(cudaStreamDestroy(stream)); check(cudaFree(weights)); check(cudaFree(checksums));
    check(cudaFree(x)); check(cudaFree(s)); check(cudaFree(y));
}
