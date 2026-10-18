#pragma once

#include <memory>
#include <string>

class AcceleratorBackend;
class Backend;
class HostMatmulBackend;

std::unique_ptr<Backend> create_cpu_backend();

enum class AcceleratorBackendKind {
    METAL,
    CUDA,
};

// Construct a compiled accelerator implementation. A null result means that
// the requested backend was not included in this build; reason describes the
// build-time limitation for the Engine fallback policy.
std::unique_ptr<AcceleratorBackend> create_accelerator_backend(
    AcceleratorBackendKind kind, std::string& reason);

// Optional Metal compute for selected host-resident matmuls.
std::unique_ptr<HostMatmulBackend> create_metal_host_matmul_backend(
    std::string& reason);

// Optional host-resident SSD MoE batch FFN delegation.
std::unique_ptr<Backend> create_metal_host_moe_ffn_backend(std::string& reason);
