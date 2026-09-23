#include "backends/factory.h"

#include "runtime/accelerator_backend.h"
#include "backends/cpu/backend.h"

#ifdef MOLLM_METAL
#include "backends/metal/backend.h"
#endif
#ifdef MOLLM_CUDA
#include "backends/cuda/backend.h"
#endif

std::unique_ptr<Backend> create_cpu_backend() {
    return std::make_unique<CPUBackend>();
}

std::unique_ptr<AcceleratorBackend> create_accelerator_backend(
    AcceleratorBackendKind kind, std::string& reason) {
    reason.clear();
    switch (kind) {
    case AcceleratorBackendKind::METAL:
#ifdef MOLLM_METAL
        return std::make_unique<MetalBackend>();
#else
        reason = "built without MOLLM_METAL";
        return nullptr;
#endif
    case AcceleratorBackendKind::CUDA:
#ifdef MOLLM_CUDA
        return std::make_unique<CudaBackend>();
#else
        reason = "built without MOLLM_CUDA";
        return nullptr;
#endif
    }
    reason = "unknown accelerator backend";
    return nullptr;
}
