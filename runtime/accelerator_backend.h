#pragma once

#include "runtime/backend.h"

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

enum class OperatorFallbackPolicy {
    ALLOW_REFERENCE,
    REQUIRE_NATIVE,
};

enum class PersistentHostAccess {
    // The complete allocation must remain directly host-addressable. This is
    // useful for low-level tooling and backends whose native storage is shared.
    FULL,
    // Only a leading metadata prefix is read on the host. Callers must update
    // it through Backend transfer/zero methods so the device copy stays in
    // sync. The prefix size is passed to alloc_persistent().
    MIRRORED_PREFIX,
    // The prefix is host-owned control metadata and device operators access
    // only the payload after it. Transfer methods expose a coherent logical
    // tensor, but discrete backends need not copy prefix-only updates to the
    // device. This avoids tiny synchronous transfers for frequently updated
    // state such as KV-cache lengths.
    HOST_AUTHORITATIVE_PREFIX,
    // The host never dereferences this allocation.
    NONE,
};

// NOTE on implementation status: these modes describe the capability a caller
// needs, not a guarantee that every backend currently differentiates them.
// A backend whose persistent storage is host-shared (Metal, via
// MTLResourceStorageModeShared) satisfies all four identically and may ignore
// host_access/host_prefix_bytes. Callers must still route every update through
// the Backend transfer/zero methods: that is correct for both a shared
// allocation and a future discrete one, and keeps storage policy a backend
// decision.

// Common lifecycle for graph-resident accelerator backends. LLMEngine only
// talks to this interface; Metal and future device backends own their
// resource representation and transfer policy behind it.
class AcceleratorBackend : public Backend {
public:
    ~AcceleratorBackend() override = default;

    virtual bool available() const = 0;

    // Prepare package/constant storage. Backends may wrap a unified host
    // region or upload individual constants.
    virtual bool register_weight_region(void* base, size_t size) = 0;
    virtual void wrap_weight(Tensor& tensor) = 0;
    virtual void wrap_weight_int4(Tensor& tensor,
                                  bool keep_native_experts = false) = 0;
    // CPU-specific repacks are optional when a backend owns every prepared
    // linear weight. Raw package layouts remain available for the explicit
    // reference fallback, so disabling sidecars changes memory/performance,
    // not serialized weight semantics.
    virtual bool wants_cpu_weight_sidecars() const { return true; }

    // Persistent state and reusable graph-boundary transfers.
    virtual void alloc_persistent(
        Tensor& tensor, size_t nbytes,
        PersistentHostAccess host_access = PersistentHostAccess::FULL,
        size_t host_prefix_bytes = 0) = 0;
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
    // Optional greedy-only boundary: keep logits on device and return just
    // their stable argmax. Backends opt in explicitly so existing execution
    // and sampling behavior is unchanged by default.
    virtual bool supports_lm_head_argmax(const Tensor&) const {
        return false;
    }
    virtual int lm_head_argmax_device_and_end_graph(
        const Tensor&, size_t, const Tensor&, int, int, int = 0,
        Tensor* = nullptr) {
        return -1;
    }
    // Optional small-batch projection used by speculative verification.
    // Backends advertise support before the engine leaves a graph open for
    // one of the device-tail variants.
    virtual bool supports_lm_head_small_batch(const Tensor&) const {
        return false;
    }
    virtual bool lm_head_small_batch(
        const float*, const Tensor&, float*, int, int, int, int = 0) {
        return false;
    }
    virtual bool lm_head_small_batch_device_and_end_graph(
        const Tensor&, const Tensor&, float*, int, int, int, int = 0) {
        return false;
    }
    virtual bool lm_head_small_batch_argmax_device_and_end_graph(
        const Tensor&, const Tensor&, int*, int, int, int, int = 0) {
        return false;
    }

    // Optional SSD-MoE hooks. They remain no-ops for accelerators that do not
    // implement direct expert streaming.
    virtual void enable_weight_copy_mode() {}
    virtual bool has_weight_copies() const { return false; }
    // Unified-memory backends may switch short SSD-MoE prefills to CPU while
    // retaining or rebuilding dense accelerator weight copies for long runs.
    virtual bool supports_moe_ssd_prefill_switching() const { return false; }
    virtual bool configure_moe_ssd_io(const std::string&, size_t, int, bool) {
        return false;
    }
    // Backends that do not override this retain their established reference
    // fallback behavior and explicitly reject native-only mode.
    virtual bool set_operator_fallback_policy(
        OperatorFallbackPolicy policy) {
        return policy == OperatorFallbackPolicy::ALLOW_REFERENCE;
    }
};
