#pragma once

#include "runtime/host_matmul_backend.h"
#include <memory>
#include <string>

// Synchronous, host-resident Q5 GEMV delegation. All Objective-C ownership
// lives in Impl; each engine gets independent buffers and weight registrations.
class MetalHostMatmulBackend final : public HostMatmulBackend {
public:
    MetalHostMatmulBackend();
    ~MetalHostMatmulBackend() override;
    bool available(std::string& reason) const;
    uint64_t completed_dispatches() const override;
    bool register_weight(const Tensor&, const void*, size_t) override;
    HostMatmulResult dispatch(const GraphNode&,
        const std::vector<const Tensor*>&, Tensor*, ThreadPool*) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
