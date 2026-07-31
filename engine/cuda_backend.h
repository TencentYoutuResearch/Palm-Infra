#pragma once

#include "engine/accelerator_backend.h"

#include <memory>

// CUDA backend with device-resident graph storage. CUDA owns prepared linear
// weights and managed intermediate/persistent buffers. Operators not yet
// implemented natively synchronize and use the CPU reference dispatcher over
// those managed buffers, providing an explicit incremental migration path.
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
    void synchronize_for_host_read() override;
    void begin_graph() override;
    void end_graph() override;
    Precision kv_cache_precision(Precision) const override {
        return Precision::FP32;
    }

    bool register_weight_region(void* base, size_t size) override;
    void wrap_weight(Tensor& tensor) override;
    void wrap_weight_int4(Tensor& tensor,
                          bool keep_native_experts = false) override;
    void alloc_persistent(Tensor& tensor, size_t nbytes) override;
    void upload_input(Tensor& tensor, const std::string& key,
                      const void* host_src, size_t nbytes) override;

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
