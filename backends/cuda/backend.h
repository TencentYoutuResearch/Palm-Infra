#pragma once

#include "runtime/accelerator_backend.h"

#include <memory>

// CUDA graph backend with device-resident graph storage. CUDA owns prepared linear
// weights and cudaMalloc-backed intermediate/persistent buffers. KV metadata
// may retain a small coherent host prefix. Operators not yet implemented
// natively use an explicit D2H -> CPU reference -> H2D bridge.
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
    void begin_execution() override;
    bool set_logical_prefill_batch(bool enabled) override;
    bool logical_prefill_batch_enabled() const override { return logical_prefill_batch_active_; }
    void prepare_graph(const Graph& graph) override;
    void release_graph_plan(const Graph& graph) override;
    bool dispatch_moe_pair(const GraphNode& node,
        const std::vector<const Tensor*>& a, const std::vector<const Tensor*>& b,
        Tensor* output_a, Tensor* output_b) override;
    int independent_stateful_mutable_input(const GraphNode& node) const override;
    uint64_t independent_stateful_mutable_inputs(const GraphNode& node) const override;
    bool supports_sigmoid_gr_reduce_fusion() const override;
    bool supports_hc_up_reduce_fusion() const override;
    int hc_up_reduce_max_rows() const override;
    bool dispatch_hc_up_reduce(const GraphNode& up,const GraphNode& reduce,
        const std::vector<const Tensor*>& inputs,Tensor* output,ThreadPool*) override;
    bool dispatch_sigmoid_gr_reduce(const GraphNode& node,
        const std::vector<const Tensor*>& inputs, Tensor* output,
        ThreadPool* thread_pool) override;
    bool dispatch_failed() const override;
    bool supports_gr_inject_rms_fusion() const override;
    bool supports_moe_gr_inject_rms_fusion() const override;
    bool dispatch_moe_gr_inject_rms(const GraphNode&,const GraphNode&,const GraphNode&,
        const std::vector<const Tensor*>&,const std::vector<const Tensor*>&,
        Tensor*,Tensor*,Tensor*,ThreadPool*) override;
    size_t moe_gr_inject_rms_fused_calls() const;
    size_t gr_inject_rms_fused_calls() const;
    size_t deferred_rms_producer_calls() const;
    size_t deferred_rms_materializations() const;
    size_t compact_carry_producer_calls() const;
    size_t compact_carry_consumer_calls() const;
    size_t compact_carry_residual_reads() const;
    size_t compact_carry_restorations() const;
    size_t compact_carry_allocation_registrations() const;
    size_t attention_head_group_calls() const;
    size_t attention_native_pv_calls() const;
    size_t attention_head_lane_calls() const;
    size_t attention_score_workspace_bytes() const;
    size_t attention_query_group_calls() const;
    size_t nvfp4_activation_fused_calls() const;
    size_t nvfp4_gate_scale_cache_bytes() const;
    size_t nvfp4_input_kmajor_calls() const;
    size_t nvfp4_token_pair_word_calls() const;
    size_t nvfp4_token_code_kmajor_calls() const;
    bool dispatch_gr_inject_rms(const GraphNode& inject, const GraphNode& rms,
        const std::vector<const Tensor*>& inputs, Tensor* carry, Tensor* output,
        ThreadPool* thread_pool) override;
    bool set_operator_fallback_policy(
        OperatorFallbackPolicy policy) override;

    void* alloc_output(Tensor& output, size_t nbytes,
                       BufferPool* pool) override;
    void free_output(Tensor& tensor, BufferPool* pool) override;
    bool is_device_resident() const override { return true; }
    bool reshape_view_strides(const Tensor& source, const Tensor& target,
                              size_t strides[4]) const override;
    size_t strided_reshape_view_calls() const;
    size_t prefill_shared_cache_bytes() const;
    size_t prefill_shared_cache_hits() const;
    bool supports_state_snapshot(const std::vector<Tensor*>& states) const override;
    bool begin_state_prefixes(const std::vector<Tensor*>& states, int tokens) override;
    bool restore_state_prefix(int kept) override;
    void end_state_prefixes() override;
    std::unique_ptr<BackendStateSnapshot> capture_state_snapshot(
        const std::vector<Tensor*>& states) override;
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
    // Explicit between-call diagnostic. No automatic policy/CLI activation.
    // Rejects active segment/capture, synchronizes work, invalidates owned
    // graph caches, then frees only returned output-pool blocks.
    bool trim_cached_outputs(size_t* reclaimed_bytes, size_t* invalidated_graphs);
    Precision kv_cache_precision(Precision requested) const override {
        return requested == Precision::FP16 || requested == Precision::FP32
            ? requested : Precision::FP32;
    }

    bool register_weight_region(void* base, size_t size) override;
    void wrap_weight(Tensor& tensor) override;
    void wrap_weight_int4(Tensor& tensor,
                          bool keep_native_experts = false) override;
    void alloc_persistent(
        Tensor& tensor, size_t nbytes,
        PersistentHostAccess host_access = PersistentHostAccess::FULL,
        size_t host_prefix_bytes = 0) override;
    void upload_input(Tensor& tensor, const std::string& key,
                      const void* host_src, size_t nbytes) override;

    bool supports_lm_head(const Tensor& weight) const override;
    bool supports_lm_head_argmax(const Tensor& weight) const override;
    bool supports_lm_head_small_batch(const Tensor& weight) const override;
    bool lm_head_small_batch_argmax_device_and_end_graph(
        const Tensor& activation, const Tensor& weight, int* output,
        int tokens, int columns, int inner, int activation_kind = 0) override;
    void lm_head_gemv(const float* activation_host, const Tensor& weight,
                      float* output_host, int n, int k,
                      int activation = 0) override;
    void lm_head_gemv_device_and_end_graph(
        const Tensor& activation, size_t activation_element_offset,
        const Tensor& weight, float* output_host, int n, int k,
        int activation_kind = 0) override;
    int lm_head_argmax_device_and_end_graph(
        const Tensor& activation, size_t activation_element_offset,
        const Tensor& weight, int n, int k, int activation_kind = 0,
        Tensor* hidden_copy = nullptr) override;

private:
    bool try_decode_segment(const GraphNode& node,
        const std::vector<const Tensor*>& inputs, Tensor* output, ThreadPool* thread_pool,
        uint64_t fusion_tag = 0, const void* fusion_weight = nullptr,
        const void* fusion_cache = nullptr);
    void finish_decode_segment();
    void abandon_decode_segment();
    bool try_nvfp4_moe(const GraphNode& node,
                       const std::vector<const Tensor*>& inputs, Tensor* output);
    bool try_fp16_moe(const GraphNode& node,
                     const std::vector<const Tensor*>& inputs, Tensor* output);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool moe_window_merge_active_ = false;
    bool logical_prefill_batch_active_ = false;
};
