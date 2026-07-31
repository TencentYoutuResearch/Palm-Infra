#pragma once

#include "engine/accelerator_backend.h"

#include <memory>

// CUDA backend with device-resident graph storage. CUDA owns prepared linear
// weights and cudaMalloc-backed intermediate/persistent buffers. KV cache
// payloads and recurrent state stay device-only; only the 64-byte KV metadata
// prefix has a host mirror maintained through explicit backend transfers.
// Operators not yet implemented natively use an explicit D2H -> CPU reference
// -> H2D bridge, providing an incremental migration path without making every
// graph allocation host-addressable.
class CudaBackend final : public AcceleratorBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    bool available() const override;
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }

    void dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* thread_pool) override;
    void clear_dispatch_error() override;
    bool dispatch_failed() const override;

    void* alloc_output(Tensor& output, size_t nbytes,
                       BufferPool* pool) override;
    void free_output(Tensor& tensor, BufferPool* pool) override;
    bool is_device_resident() const override { return true; }
    bool copy_to_host(const Tensor& source, void* destination,
                      size_t nbytes, size_t source_offset = 0) override;
    bool copy_from_host(const void* source, Tensor& destination,
                        size_t nbytes,
                        size_t destination_offset = 0) override;
    bool zero_tensor(Tensor& tensor, size_t nbytes,
                     size_t destination_offset = 0) override;
    bool round_to_bf16(Tensor& tensor) override;
    void synchronize_for_host_read() override;
    void begin_graph() override;
    void end_graph() override;
    Precision kv_cache_precision(Precision requested) const override {
        return requested;
    }

    bool register_weight_region(void* base, size_t size) override;
    void wrap_weight(Tensor& tensor) override;
    void wrap_weight_int4(Tensor& tensor,
                          bool keep_native_experts = false) override;
    bool wants_cpu_weight_sidecars() const override { return false; }
    void alloc_persistent(
        Tensor& tensor, size_t nbytes,
        PersistentHostAccess host_access = PersistentHostAccess::FULL,
        size_t host_prefix_bytes = 0) override;
    void upload_input(Tensor& tensor, const std::string& key,
                      const void* host_src, size_t nbytes) override;
    bool configure_moe_device_cache(size_t capacity_bytes) override;
    DeviceMoeCacheStats moe_device_cache_stats() const override;
    BackendOperatorStats operator_stats() const override;
    bool set_operator_fallback_policy(
        OperatorFallbackPolicy policy) override;

    bool supports_lm_head(const Tensor& weight) const override;
    void lm_head_gemv(const float* activation_host, const Tensor& weight,
                      float* output_host, int n, int k,
                      int activation = 0) override;
    void lm_head_gemv_device_and_end_graph(
        const Tensor& activation, size_t activation_element_offset,
        const Tensor& weight, float* output_host, int n, int k,
        int activation_kind = 0) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
