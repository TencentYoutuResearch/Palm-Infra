#pragma once

#include "engine/backend.h"

#include <cstdint>
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

struct DeviceMoeCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t host_to_device_bytes = 0;
    uint64_t fallback_host_to_device_bytes = 0;
    // Logical payload bytes made available to kernels without staging copies.
    uint64_t direct_expert_bytes = 0;
    size_t resident_bytes = 0;
    size_t capacity_bytes = 0;
    size_t peak_selected_bytes = 0;
    size_t fallback_scratch_bytes = 0;
};

// Aggregate graph-dispatch coverage reported by accelerator backends. A
// tracked backend classifies every dispatched node as either native or an
// explicit reference-backend fallback. This remains separate from timing
// profiles so correctness tests and embedding applications do not need to
// enable or scrape diagnostic logging.
struct BackendOperatorStats {
    bool tracked = false;
    uint64_t native_calls = 0;
    uint64_t fallback_calls = 0;
};

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
    // The host never dereferences this allocation.
    NONE,
};

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

    // Optional SSD-MoE hooks. They remain no-ops for accelerators that do not
    // implement direct expert streaming.
    virtual void enable_weight_copy_mode() {}
    virtual bool has_weight_copies() const { return false; }
    virtual bool configure_moe_ssd_io(const std::string&, size_t, int, bool) {
        return false;
    }
    virtual bool configure_moe_device_cache(size_t capacity_bytes) {
        return capacity_bytes == 0;
    }
    virtual DeviceMoeCacheStats moe_device_cache_stats() const { return {}; }
    virtual BackendOperatorStats operator_stats() const { return {}; }
    // Backends that do not override this retain their established reference
    // fallback behavior and explicitly reject native-only mode.
    virtual bool set_operator_fallback_policy(
        OperatorFallbackPolicy policy) {
        return policy == OperatorFallbackPolicy::ALLOW_REFERENCE;
    }
};
