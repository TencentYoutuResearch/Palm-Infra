#pragma once

#include "runtime/expert_provider.h"

#include <vector>

class ThreadPool;

// Optional synchronous FFN for a batch of already-borrowed SSD experts.
// False leaves the caller free to compute the entire batch on the CPU.
class HostMoeFfn {
public:
    virtual ~HostMoeFfn() = default;
    virtual bool run_ready_batch(ExpertProvider& provider,
                                 const std::vector<ExpertLease>& leases,
                                 const float* input, int hidden, int intermediate,
                                 float swiglu_limit, float* output,
                                 ThreadPool* pool) = 0;
};
