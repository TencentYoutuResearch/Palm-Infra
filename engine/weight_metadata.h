#pragma once

#include "graph/mmap_file.h"
#include "kernels/tensor.h"

namespace mollm::detail {

/// Validate serialized quantization metadata against the graph-provided tensor
/// shape, then attach it to the tensor. Non-quantized tensors are accepted
/// after their quantization fields are cleared.
bool configure_weight_metadata(Tensor& tensor,
                               const MappedFile::Header& header,
                               const void* scales, const char* label);

}  // namespace mollm::detail
