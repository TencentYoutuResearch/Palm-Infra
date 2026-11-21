#pragma once

#include "kernels/cuda/matmul.h"
#include "kernels/cuda/elementwise.h"

#include <cstdint>
#include <cstddef>

namespace mollm_cuda {
// Private approximate draft-head cache; original Q8 remains the target head.
cudaError_t pack_q8_draft_head_compact(
    const int8_t* weights, const float* scales, Q4CompactG32* output,
    int rows, int inner, cudaStream_t stream);
// Load-time least-squares Q4 packing. Same transient Q4CompactG32 layout and
// decoder as the max-based packer above; original Q8 weights remain immutable.
cudaError_t pack_q8_draft_head_compact_fitted(
    const int8_t* weights, const float* scales, Q4CompactG32* output,
    int rows, int inner, cudaStream_t stream);
// Argmax is produced first in the same stream. Scratch needs groups floats;
// result shares its first eight bytes with the existing ArgMaxPair output.
struct DraftHeadResult {
    ArgMaxPair top1;
    float probability;
    float reserved;
};
static_assert(sizeof(DraftHeadResult) == 16);
cudaError_t launch_draft_head_probability(
    const float* logits, int count, DraftHeadResult* result,
    float* scratch, int groups, cudaStream_t stream);

// Approximate draft-only vocabulary screening; target Q8 remains authoritative.
// Qualified geometry: N248320/K2560. Select 32 (default), 24 or 16 K32 groups
// by input energy,
// rank the vocabulary, then rescore 1024 candidates with the original FP32
// chains/tree. Other output logits are -infinity, so their probability mass is
// omitted. The existing probability API therefore returns an approximation.
struct DraftHeadScreenScratch {
    float* coarse = nullptr;       // N floats
    float* sorted = nullptr;       // N floats
    int* ids = nullptr;            // N ints, initialized once
    int* sorted_ids = nullptr;     // N ints
    int* groups = nullptr;         // 32 ints, recomputed per input
    void* sort_temp = nullptr;     // 256-byte aligned
    size_t sort_bytes = 0;
};
// All scratch spans, input, output and weights must be disjoint. Query needs
// the four vocabulary arrays and groups, but no sort_temp allocation yet.
cudaError_t draft_head_screen_temp_bytes(
    const DraftHeadScreenScratch& scratch, size_t& bytes);
cudaError_t initialize_draft_head_screen(
    const DraftHeadScreenScratch& scratch, cudaStream_t stream);
cudaError_t launch_draft_head_screen(const float* input,
    const Q4CompactG32* weights, float* output, int rows, int inner,
    const DraftHeadScreenScratch& scratch, cudaStream_t stream, int group_count = 32);
}  // namespace mollm_cuda
