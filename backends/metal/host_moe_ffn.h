#pragma once

#include "runtime/backend.h"
#include "runtime/host_moe_ffn.h"
#include <memory>
#include <string>

// Synchronous Metal delegation of eligible, already-resident IQ2/IQ3 MoE
// batches. The surrounding route, shared expert, and scatter stay on CPU.
class MetalHostMoeFfn final : public Backend, public HostMoeFfn {
public:
    MetalHostMoeFfn();
    ~MetalHostMoeFfn() override;
    bool available(std::string& reason) const;
    uint64_t completed_host_moe_batches() const override;
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }
    void dispatch(const GraphNode&, const std::vector<const Tensor*>&,
                  Tensor*, ThreadPool*) override;
    bool dispatch_host_moe(const GraphNode&, const std::vector<const Tensor*>&,
                           Tensor*, ThreadPool*, bool& success) override;
    bool run_ready_batch(ExpertProvider& provider,
                         const std::vector<ExpertLease>& leases,
                         const float* input, int hidden, int intermediate,
                         float swiglu_limit, float* output,
                         ThreadPool* pool) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
