#pragma once

#include "runtime/backend.h"
#include "kernels/cpu_platform.h"

// CPU graph dispatcher. The implementation is in backends/cpu/backend.cpp.
class CPUBackend : public Backend {
public:
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }

    void dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* thread_pool) override;

    void clear_dispatch_error() override { dispatch_failed_ = false; }
    bool dispatch_failed() const override { return dispatch_failed_; }
    Precision kv_cache_precision(Precision requested) const override {
        if (requested == Precision::FP16 &&
            !mollm::cpu::capabilities().fp16_kv_cache)
            return Precision::FP32;
        return requested;
    }

private:
    bool dispatch_failed_ = false;
};
