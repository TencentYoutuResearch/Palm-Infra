#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace mollm_cuda {
// Exact greedy-head screen: a packed approximation bounds every row score, and
// candidate rows are rescored with the original Q8 weights. Fixed K=2560,
// per-row Q8 scales.
// Keep original Q8 weights: this auxiliary packed copy is not a model format.
struct Q8HeadScreenScratch {
    float* approximate; // rows
    float* lower;       // 257
    double* norm;       // 1
    unsigned* count;    // 1
    int* indices;       // rows
};
cudaError_t pack_q8_head_screen(const int8_t* weights, uint8_t* packed,
    int rows, cudaStream_t stream);
// Output is sparse exact Q8: discarded rows are -infinity. ONLY argmax may
// consume it; never use for sampling, probabilities, or returned full logits.
// Requires 16-byte-aligned input; all scratch allocations are caller-owned.
cudaError_t launch_q8_head_screen(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int rows,
    Q8HeadScreenScratch scratch, cudaStream_t stream);
// Scalar Q4 uses 1280 packed bytes per row (K=2560), distinct from the
// 1920-byte Q6 format consumed by the batch APIs below. Exact original-Q8
// scores from two anchors tighten conservative intervals; all surviving rows
// are exactly rescored, without a candidate cap. Requires 512 <= rows <=
// INT32_MAX-65536: indices[0..511] temporarily hold anchor IDs before selection.
// Input is 16-byte aligned; other arrays require their natural alignment and
// disjoint storage.
cudaError_t pack_q8_head_screen_q4(const int8_t* weights, uint8_t* packed,
    int rows, cudaStream_t stream);
cudaError_t launch_q8_head_screen_q4(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int rows,
    Q8HeadScreenScratch scratch, cudaStream_t stream);
// Three contiguous K=2560 queries sharing the same packed weight copy.
// Separate scratch type prevents accidentally passing single-query capacity.
struct Q8HeadScreenBatch3Scratch {
    float* approximate; // 3 * rows, query-major
    float* lower;       // 3 * 257
    double* norm;       // 3
    unsigned* count;    // 3
    int* indices;       // 3 * rows, query-major
};
// Same sparse-argmax-only contract as launch_q8_head_screen. Input contains
// 3 * 2560 floats; output contains 3 * rows floats. Packed weights are read-only.
cudaError_t launch_q8_head_screen_batch3(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int rows,
    Q8HeadScreenBatch3Scratch scratch, cudaStream_t stream);
// Four to eight contiguous K=2560 queries. Capacities count elements, not bytes;
// separate fields reject a scalar/three-query workspace before launching work.
// Arrays must not overlap; float/unsigned/int arrays require 4-byte alignment,
// norm requires 8-byte alignment. Input and output have queries * 2560 and
// queries * rows elements. The original rows * 1920-byte packed copy is read-only.
struct Q8HeadScreenBatchScratch {
    float* approximate;
    float* lower;
    double* norm;
    unsigned* count;
    int* indices;
    size_t approximate_capacity, lower_capacity, norm_capacity;
    size_t count_capacity, indices_capacity;
};
// Same sparse exact Q8, argmax-only output contract as the scalar/Batch3 APIs.
cudaError_t launch_q8_head_screen_batch(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int queries, int rows,
    Q8HeadScreenBatchScratch scratch, cudaStream_t stream);

}  // namespace mollm_cuda
