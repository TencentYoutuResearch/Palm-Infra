#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <memory>
#include <string>

// Owns package-backed Metal views, persistent tensors, and reusable boundary
// input buffers. This header stays plain C++; Objective-C ownership is hidden
// in resource_store.mm.
class MetalResourceStore {
public:
    explicit MetalResourceStore(void* device);
    ~MetalResourceStore();

    MetalResourceStore(const MetalResourceStore&) = delete;
    MetalResourceStore& operator=(const MetalResourceStore&) = delete;

    bool register_weight_region(void* base, size_t size);
    void enable_weight_copy_mode();
    bool has_weight_copies() const;

    void wrap_weight(Tensor& tensor);
    void wrap_weight_int8(Tensor& tensor);
    void wrap_weight_int4(Tensor& tensor, bool keep_native_experts);
    void alloc_persistent(Tensor& tensor, size_t bytes);
    void upload_input(Tensor& tensor, const std::string& key,
                      const void* source, size_t bytes);
    void upload_zero_input(Tensor& tensor, const std::string& key,
                           size_t bytes);

    // Resolve a package pointer to its actual Metal buffer. This also handles
    // packages split across maxBufferLength-sized regions.
    bool locate_weight(const void* pointer, size_t bytes,
                       void*& buffer, size_t& offset) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
