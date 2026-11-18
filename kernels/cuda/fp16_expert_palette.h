#pragma once
#include <cuda_runtime.h>

namespace mollm_cuda {
struct Nvfp4Projection;
// Each job points to rows*inner byte codes and (rows/128)*(inner/128)*256
// FP16 palette entries in block_scales. Inputs/outputs are FP32. Palettes
// restart at each expert; dimensions must be multiples of 128. The caller
// owns all device extents and keeps them alive through execution/replay.
cudaError_t launch_fp16_palette_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);
}
