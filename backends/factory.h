#pragma once

#include <memory>
#include <string>

class AcceleratorBackend;
class Backend;

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
