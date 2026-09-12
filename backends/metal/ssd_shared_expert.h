#pragma once

#include "kernels/tensor.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>

class MetalBufferPool;
class MetalPipelineCache;

// Runs the dense shared-expert branch alongside SSD routed-expert I/O on an
// independent Metal command stream. The caller retains scheduling ownership
// and later waits on event()/Work::ready_value before combining the outputs.
class MetalSsdSharedExpert {
public:
    struct Work {
        void* qx = nullptr;
        void* sx = nullptr;
        void* intermediate = nullptr;
        void* qintermediate = nullptr;
        void* qintermediate_scale = nullptr;
        void* scale = nullptr;
        void* output = nullptr;
        size_t qx_bytes = 0;
        size_t sx_bytes = 0;
        size_t intermediate_bytes = 0;
        size_t qintermediate_bytes = 0;
        size_t output_bytes = 0;
        uint64_t ready_value = 0;
    };

    MetalSsdSharedExpert(void* device, MetalBufferPool* pool,
                         MetalPipelineCache* pipelines);
    ~MetalSsdSharedExpert();

    MetalSsdSharedExpert(const MetalSsdSharedExpert&) = delete;
    MetalSsdSharedExpert& operator=(const MetalSsdSharedExpert&) = delete;

    bool configure();
    bool submit(const Tensor& x, const Tensor& gate, const Tensor& up,
                const Tensor& down, const Tensor& scale_weight, int hidden,
                int intermediate, int seq, int layer, Work& work);

    id<MTLSharedEvent> event() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
