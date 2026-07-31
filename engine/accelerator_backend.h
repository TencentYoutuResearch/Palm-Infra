#pragma once

#include "engine/backend.h"

#include <string>
#include <string_view>

namespace mollm::detail {

inline bool is_routed_expert_aggregate_ref(std::string_view reference) {
    const bool expert =
        reference.find("_experts_") != std::string_view::npos ||
        reference.find(".experts.") != std::string_view::npos;
    const bool shared =
        reference.find("_shared_experts_") != std::string_view::npos ||
        reference.find(".shared_experts.") != std::string_view::npos;
    return expert && !shared;
}

}  // namespace mollm::detail

// Common lifecycle for graph-resident accelerator backends. LLMEngine only
// talks to this interface; Metal, CUDA and future device backends own their
// resource representation and transfer policy behind it.
class AcceleratorBackend : public Backend {
public:
    ~AcceleratorBackend() override = default;

    virtual bool available() const = 0;

    // Prepare package/constant storage. Backends may wrap a unified host
    // region (Metal) or upload individual constants (CUDA).
    virtual bool register_weight_region(void* base, size_t size) = 0;
    virtual void wrap_weight(Tensor& tensor) = 0;
    virtual void wrap_weight_int4(Tensor& tensor,
                                  bool keep_native_experts = false) = 0;

    // Persistent state and reusable graph-boundary transfers.
    virtual void alloc_persistent(Tensor& tensor, size_t nbytes) = 0;
    virtual void upload_input(Tensor& tensor, const std::string& key,
                              const void* host_src, size_t nbytes) = 0;

    // Host logits remain the sampler boundary for now.
    virtual bool supports_lm_head(const Tensor& weight) const = 0;
    virtual void lm_head_gemv(const float* activation_host,
                              const Tensor& weight, float* output_host,
                              int n, int k, int activation = 0) = 0;
    virtual void lm_head_gemv_device_and_end_graph(
        const Tensor& activation, size_t activation_element_offset,
        const Tensor& weight, float* output_host, int n, int k,
        int activation_kind = 0) = 0;

    // Optional SSD-MoE hooks. They remain no-ops for accelerators that do not
    // implement direct expert streaming.
    virtual void enable_weight_copy_mode() {}
    virtual bool has_weight_copies() const { return false; }
    virtual bool configure_moe_ssd_io(const std::string&, size_t, int, bool) {
        return false;
    }
};
