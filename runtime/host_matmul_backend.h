#pragma once

#include "core/tensor.h"
#include <cstddef>
#include <vector>

struct GraphNode;
class ThreadPool;

enum class HostMatmulResult { Unsupported, Completed, Failed };

// Optional device execution of selected matmuls on host-owned graph tensors.
// This interface does not allocate KV caches or change graph storage ownership.
class HostMatmulBackend {
public:
    virtual ~HostMatmulBackend() = default;
    // Successful device dispatches since construction (not registrations).
    virtual uint64_t completed_dispatches() const { return 0; }

    // The caller selects eligible constants and owns their mapping. Destroy
    // this backend before unmapping it. Reject ranges that cannot be borrowed.
    virtual bool register_weight(const Tensor& weight,
                                 const void* mapping, size_t mapping_bytes) = 0;
    virtual HostMatmulResult dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* pool) = 0;
};
