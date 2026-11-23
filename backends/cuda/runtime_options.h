#pragma once

// Every environment variable the CUDA backend reads, parsed once per backend
// instance. A backend reads them when it is constructed, so set them first.

#include "backends/cuda/prefill_batch_policy.h"

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace mollm_cuda {

// An override of a default the backend otherwise derives from the request.
enum class Override { Unset = -1, Off = 0, On = 1 };

inline bool resolve(Override option, bool fallback) {
    return option == Override::Unset ? fallback : option == Override::On;
}

struct RuntimeOptions {
    // Opt-outs (set to any value).
    bool disable_qualified_verification = false; // MOLLM_CUDA_DISABLE_QUALIFIED_VERIFICATION
    bool disable_decode_segments = false;        // MOLLM_CUDA_DISABLE_DECODE_SEGMENTS
    bool disable_geometric_growth = false;       // MOLLM_CUDA_DISABLE_GEOMETRIC_GROWTH
    bool disable_nvfp4_moe = false;              // MOLLM_CUDA_DISABLE_NVFP4_MOE
    bool disable_nvfp4_grouping = false;         // MOLLM_CUDA_DISABLE_NVFP4_GROUPING
    bool disable_resident_routes = false;        // MOLLM_CUDA_DISABLE_RESIDENT_ROUTES
    bool disable_moe_graphs = false;             // MOLLM_CUDA_DISABLE_MOE_GRAPHS
    // Leave a demand-cached MoE call with a missing expert to the CPU kernel
    // instead of requesting its experts and waiting for them on the GPU path.
    bool disable_moe_demand_wait = false;        // MOLLM_CUDA_DISABLE_MOE_DEMAND_WAIT

    // Opt-ins (exactly "1").
    bool dense_prefill_fp32_reference = false; // MOLLM_CUDA_DENSE_PREFILL_FP32_REFERENCE
    bool q4_small_batch = false;               // MOLLM_CUDA_Q4_SMALL_BATCH
    bool nvfp4_batch_pipeline = false;         // MOLLM_CUDA_NVFP4_BATCH_PIPELINE
    bool nvfp4_mma_prepare_singles = false;    // MOLLM_CUDA_NVFP4_MMA_PREPARE_SINGLES
    bool nvfp4_down_row_scale_prefetch = false; // MOLLM_CUDA_NVFP4_DOWN_ROW_SCALE_PREFETCH

    // On unless set to something other than "1".
    bool qsa_gqa_reuse = true;    // MOLLM_CUDA_QSA_GQA_REUSE
    bool qsa_decode_split = true; // MOLLM_CUDA_QSA_DECODE_SPLIT

    // Overrides: "1" enables, any other value disables, unset keeps the default.
    Override strided_reshape_views = Override::Unset;            // MOLLM_CUDA_STRIDED_RESHAPE_VIEWS
    Override moe_prefill_input_half_reuse = Override::Unset;     // MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE
    Override nvfp4_prefill_sort = Override::Unset; // MOLLM_CUDA_NVFP4_PREFILL_SORT, else MOLLM_CUDA_NVFP4_MMA_PREFILL_SORT
    Override nvfp4_prefill_input_scale_kmajor = Override::Unset; // MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR
    Override nvfp4_prefill_compact_logits = Override::Unset;     // MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS
    Override nvfp4_prefill_token_quantize = Override::Unset;     // MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE
    Override nvfp4_prefill_token_pair_words = Override::Unset;   // MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS
    Override nvfp4_prefill_token_codes_kmajor = Override::Unset; // MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR
    Override sdpa_prefill_validated_prefix = Override::Unset;    // MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX
    Override sdpa_prefill_skip_zero_pv = Override::Unset;        // MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV
    Override sdpa_prefill_pv_pair = Override::Unset;             // MOLLM_CUDA_SDPA_PREFILL_PV_PAIR

    // Values.
    int q4_row_major = 0;        // MOLLM_CUDA_Q4_ROW_MAJOR: 0, 1 or 2.
    int nvfp4_mma_gate_parts = 1; // MOLLM_CUDA_NVFP4_MMA_GATE_PARTS: 1, 2, 4 or 8.
    size_t prefill_fp16_cache_bytes = 0;       // MOLLM_CUDA_PREFILL_FP16_CACHE_MB, at most 32768.
    size_t prefill_gate_scale_cache_bytes = 0; // MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB, at most 8192.
    bool output_pool_invalid = false; // MOLLM_CUDA_OUTPUT_POOL_MB was set but is not a size.
    size_t output_pool_limit_bytes = 0; // MOLLM_CUDA_OUTPUT_POOL_MB; zero is unlimited.

    // Diagnostics.
    bool profile = false;                      // MOLLM_CUDA_PROFILE (set)
    bool capture_audit = false;                // MOLLM_CUDA_CAPTURE_AUDIT (set)
    bool q4_palette_audit = false;             // MOLLM_CUDA_Q4_PALETTE_AUDIT (set)
    bool trace_kernels = false;                // MOLLM_CUDA_TRACE_KERNELS (set)
    bool nvfp4_trace_pairs = false;            // MOLLM_CUDA_NVFP4_TRACE_PAIRS (set)
    bool nvfp4_trace_first_routes = false;     // MOLLM_CUDA_NVFP4_TRACE_FIRST_ROUTES ("1")
    bool nvfp4_trace_batch_occupancy = false;  // MOLLM_CUDA_NVFP4_TRACE_BATCH_OCCUPANCY ("1")
    bool sdpa_prefix_trace_validation = false; // MOLLM_CUDA_SDPA_PREFIX_TRACE_VALIDATION (set)
};

inline RuntimeOptions read_runtime_options() {
    const auto set = [](const char* name) { return std::getenv(name) != nullptr; };
    const auto one = [](const char* name) {
        const char* value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    };
    const auto unless_other = [](const char* name) {
        const char* value = std::getenv(name);
        return !value || std::strcmp(value, "1") == 0;
    };
    const auto override_of = [](const char* value) {
        if (!value) return Override::Unset;
        return logical_prefill_option(value, false) ? Override::On : Override::Off;
    };
    const auto megabytes = [](const char* name, unsigned long long limit, bool leading_digit) {
        const char* value = std::getenv(name);
        if (!value || !*value) return size_t{0};
        if (leading_digit && (*value < '0' || *value > '9')) return size_t{0};
        char* end = nullptr;
        const unsigned long long mb = std::strtoull(value, &end, 10);
        return end && !*end && mb <= limit ? static_cast<size_t>(mb) * 1024 * 1024 : size_t{0};
    };

    RuntimeOptions options;
    options.disable_qualified_verification = set("MOLLM_CUDA_DISABLE_QUALIFIED_VERIFICATION");
    options.disable_decode_segments = set("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS");
    options.disable_geometric_growth = set("MOLLM_CUDA_DISABLE_GEOMETRIC_GROWTH");
    options.disable_nvfp4_moe = set("MOLLM_CUDA_DISABLE_NVFP4_MOE");
    options.disable_nvfp4_grouping = set("MOLLM_CUDA_DISABLE_NVFP4_GROUPING");
    options.disable_resident_routes = set("MOLLM_CUDA_DISABLE_RESIDENT_ROUTES");
    options.disable_moe_graphs = set("MOLLM_CUDA_DISABLE_MOE_GRAPHS");
    options.disable_moe_demand_wait = set("MOLLM_CUDA_DISABLE_MOE_DEMAND_WAIT");

    options.dense_prefill_fp32_reference = one("MOLLM_CUDA_DENSE_PREFILL_FP32_REFERENCE");
    options.q4_small_batch = one("MOLLM_CUDA_Q4_SMALL_BATCH");
    options.nvfp4_batch_pipeline = one("MOLLM_CUDA_NVFP4_BATCH_PIPELINE");
    options.nvfp4_mma_prepare_singles = one("MOLLM_CUDA_NVFP4_MMA_PREPARE_SINGLES");
    options.nvfp4_down_row_scale_prefetch = one("MOLLM_CUDA_NVFP4_DOWN_ROW_SCALE_PREFETCH");

    options.qsa_gqa_reuse = unless_other("MOLLM_CUDA_QSA_GQA_REUSE");
    options.qsa_decode_split = unless_other("MOLLM_CUDA_QSA_DECODE_SPLIT");

    options.strided_reshape_views = override_of(std::getenv("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS"));
    options.moe_prefill_input_half_reuse = override_of(std::getenv("MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE"));
    const char* sort = std::getenv("MOLLM_CUDA_NVFP4_PREFILL_SORT");
    if (!sort) sort = std::getenv("MOLLM_CUDA_NVFP4_MMA_PREFILL_SORT");
    options.nvfp4_prefill_sort = override_of(sort);
    options.nvfp4_prefill_input_scale_kmajor = override_of(std::getenv("MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR"));
    options.nvfp4_prefill_compact_logits = override_of(std::getenv("MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS"));
    options.nvfp4_prefill_token_quantize = override_of(std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE"));
    options.nvfp4_prefill_token_pair_words = override_of(std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS"));
    options.nvfp4_prefill_token_codes_kmajor = override_of(std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR"));
    options.sdpa_prefill_validated_prefix = override_of(std::getenv("MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX"));
    options.sdpa_prefill_skip_zero_pv = override_of(std::getenv("MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV"));
    options.sdpa_prefill_pv_pair = override_of(std::getenv("MOLLM_CUDA_SDPA_PREFILL_PV_PAIR"));

    if (const char* value = std::getenv("MOLLM_CUDA_Q4_ROW_MAJOR"))
        options.q4_row_major = std::strcmp(value, "2") == 0 ? 2 : std::strcmp(value, "1") == 0 ? 1 : 0;
    if (const char* value = std::getenv("MOLLM_CUDA_NVFP4_MMA_GATE_PARTS"))
        options.nvfp4_mma_gate_parts = std::strcmp(value, "2") == 0 ? 2 :
            std::strcmp(value, "4") == 0 ? 4 : std::strcmp(value, "8") == 0 ? 8 : 1;
    options.prefill_fp16_cache_bytes = megabytes("MOLLM_CUDA_PREFILL_FP16_CACHE_MB", 32768, false);
    options.prefill_gate_scale_cache_bytes = megabytes("MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB", 8192, true);
    if (const char* value = std::getenv("MOLLM_CUDA_OUTPUT_POOL_MB")) {
        bool digits = *value != '\0';
        for (const char* c = value; *c; ++c) digits = digits && *c >= '0' && *c <= '9';
        errno = 0;
        char* end = nullptr;
        const unsigned long long mb = std::strtoull(value, &end, 10);
        if (!digits || errno || !end || *end || mb > std::numeric_limits<size_t>::max() / (1024 * 1024))
            options.output_pool_invalid = true;
        else
            options.output_pool_limit_bytes = static_cast<size_t>(mb) * 1024 * 1024;
    }

    options.profile = set("MOLLM_CUDA_PROFILE");
    options.capture_audit = set("MOLLM_CUDA_CAPTURE_AUDIT");
    options.q4_palette_audit = set("MOLLM_CUDA_Q4_PALETTE_AUDIT");
    options.trace_kernels = set("MOLLM_CUDA_TRACE_KERNELS");
    options.nvfp4_trace_pairs = set("MOLLM_CUDA_NVFP4_TRACE_PAIRS");
    options.nvfp4_trace_first_routes = one("MOLLM_CUDA_NVFP4_TRACE_FIRST_ROUTES");
    options.nvfp4_trace_batch_occupancy = one("MOLLM_CUDA_NVFP4_TRACE_BATCH_OCCUPANCY");
    options.sdpa_prefix_trace_validation = set("MOLLM_CUDA_SDPA_PREFIX_TRACE_VALIDATION");
    return options;
}

}  // namespace mollm_cuda
