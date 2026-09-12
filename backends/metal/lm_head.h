#pragma once

#include <cstddef>

struct Tensor;
class MetalBufferPool;
class MetalCommandContext;
class MetalPipelineCache;

// Owns Metal command encoding for vocabulary projection and optional device
// argmax tails. It borrows backend resource components and may finish the
// currently open graph command stream.
class MetalLmHead {
public:
    MetalLmHead(MetalBufferPool* pool, MetalCommandContext* commands,
                MetalPipelineCache* pipelines, bool& dispatch_failed);

    void gemv(const float* activation_host, const Tensor& weight,
              float* output_host, int n, int k, int activation);
    bool small_batch(const float* activation_host, const Tensor& weight,
                     float* output_host, int m, int n, int k, int activation);
    bool small_batch_device_and_end_graph(const Tensor& activation,
                                          const Tensor& weight,
                                          float* output_host, int m, int n,
                                          int k, int activation_kind);
    bool small_batch_argmax_device_and_end_graph(const Tensor& activation,
                                                 const Tensor& weight,
                                                 int* top1_output, int m, int n,
                                                 int k, int activation_kind);
    void gemv_device_and_end_graph(const Tensor& activation,
                                   size_t activation_element_offset,
                                   const Tensor& weight, float* output_host,
                                   int n, int k, int activation_kind);
    int argmax_device_and_end_graph(const Tensor& activation,
                                    size_t activation_element_offset,
                                    const Tensor& weight, int n, int k,
                                    int activation_kind, Tensor* hidden_copy);

private:
    void gemv_impl(void* activation_device, size_t activation_byte_offset,
                   const Tensor& weight, float* output_host, int n, int k,
                   int activation_kind, bool finish_open_graph,
                   int* top1_output = nullptr, Tensor* hidden_copy = nullptr);
    bool small_batch_impl(void* activation_device,
                          size_t activation_byte_offset, const Tensor& weight,
                          float* output_host, int m, int n, int k,
                          int activation_kind, bool finish_open_graph,
                          int* top1_output = nullptr);

    MetalBufferPool* pool_ = nullptr;
    MetalCommandContext* commands_ = nullptr;
    MetalPipelineCache* pipelines_ = nullptr;
    bool& dispatch_failed_;
};
