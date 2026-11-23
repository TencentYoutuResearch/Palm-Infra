// The CUDA backend's environment options, parsed without a CUDA toolkit.
#include "backends/cuda/runtime_options.h"

#include <cstdio>
#include <cstdlib>

using namespace mollm_cuda;

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

const char* const names[] = {
    "MOLLM_CUDA_DISABLE_QUALIFIED_VERIFICATION", "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS",
    "MOLLM_CUDA_DISABLE_GEOMETRIC_GROWTH", "MOLLM_CUDA_DISABLE_NVFP4_MOE",
    "MOLLM_CUDA_DISABLE_NVFP4_GROUPING", "MOLLM_CUDA_DISABLE_RESIDENT_ROUTES",
    "MOLLM_CUDA_DISABLE_MOE_GRAPHS", "MOLLM_CUDA_DISABLE_MOE_DEMAND_WAIT",
    "MOLLM_CUDA_DENSE_PREFILL_FP32_REFERENCE",
    "MOLLM_CUDA_Q4_SMALL_BATCH", "MOLLM_CUDA_NVFP4_BATCH_PIPELINE",
    "MOLLM_CUDA_NVFP4_MMA_PREPARE_SINGLES", "MOLLM_CUDA_NVFP4_DOWN_ROW_SCALE_PREFETCH",
    "MOLLM_CUDA_QSA_GQA_REUSE", "MOLLM_CUDA_QSA_DECODE_SPLIT", "MOLLM_CUDA_STRIDED_RESHAPE_VIEWS",
    "MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE", "MOLLM_CUDA_NVFP4_PREFILL_SORT",
    "MOLLM_CUDA_NVFP4_MMA_PREFILL_SORT", "MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR",
    "MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS", "MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE",
    "MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS", "MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR",
    "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX", "MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV",
    "MOLLM_CUDA_SDPA_PREFILL_PV_PAIR", "MOLLM_CUDA_Q4_ROW_MAJOR", "MOLLM_CUDA_NVFP4_MMA_GATE_PARTS",
    "MOLLM_CUDA_PREFILL_FP16_CACHE_MB", "MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB",
    "MOLLM_CUDA_OUTPUT_POOL_MB", "MOLLM_CUDA_PROFILE", "MOLLM_CUDA_CAPTURE_AUDIT",
    "MOLLM_CUDA_Q4_PALETTE_AUDIT", "MOLLM_CUDA_TRACE_KERNELS", "MOLLM_CUDA_NVFP4_TRACE_PAIRS",
    "MOLLM_CUDA_NVFP4_TRACE_FIRST_ROUTES", "MOLLM_CUDA_NVFP4_TRACE_BATCH_OCCUPANCY",
    "MOLLM_CUDA_SDPA_PREFIX_TRACE_VALIDATION",
};

void clear() {
    for (const char* name : names) unsetenv(name);
}

RuntimeOptions with(const char* name, const char* value) {
    clear();
    setenv(name, value, 1);
    return read_runtime_options();
}

constexpr size_t mb = 1024 * 1024;

}  // namespace

int main() {
    clear();
    const RuntimeOptions defaults = read_runtime_options();
    expect(!defaults.disable_qualified_verification && !defaults.disable_decode_segments &&
           !defaults.disable_nvfp4_moe && !defaults.disable_moe_graphs &&
           !defaults.disable_moe_demand_wait, "opt-outs default off");
    expect(with("MOLLM_CUDA_DISABLE_MOE_DEMAND_WAIT", "1").disable_moe_demand_wait, "demand wait opt-out");
    expect(!defaults.q4_small_batch && !defaults.nvfp4_batch_pipeline &&
           !defaults.dense_prefill_fp32_reference, "opt-ins default off");
    expect(defaults.qsa_gqa_reuse && defaults.qsa_decode_split, "QSA reuse defaults on");
    expect(defaults.strided_reshape_views == Override::Unset &&
           defaults.nvfp4_prefill_sort == Override::Unset &&
           defaults.sdpa_prefill_pv_pair == Override::Unset, "overrides default unset");
    expect(defaults.q4_row_major == 0 && defaults.nvfp4_mma_gate_parts == 1 &&
           defaults.prefill_fp16_cache_bytes == 0 && defaults.prefill_gate_scale_cache_bytes == 0 &&
           defaults.output_pool_limit_bytes == 0 && !defaults.output_pool_invalid, "values default");
    expect(!defaults.trace_kernels && !defaults.profile, "diagnostics default off");

    // Opt-outs react to any value, opt-ins to exactly "1".
    expect(with("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS", "0").disable_decode_segments, "opt-out by presence");
    expect(with("MOLLM_CUDA_TRACE_KERNELS", "").trace_kernels, "diagnostic by presence");
    expect(with("MOLLM_CUDA_Q4_SMALL_BATCH", "1").q4_small_batch &&
           !with("MOLLM_CUDA_Q4_SMALL_BATCH", "true").q4_small_batch &&
           !with("MOLLM_CUDA_Q4_SMALL_BATCH", "0").q4_small_batch, "opt-in by exact value");
    expect(with("MOLLM_CUDA_QSA_GQA_REUSE", "1").qsa_gqa_reuse &&
           !with("MOLLM_CUDA_QSA_GQA_REUSE", "0").qsa_gqa_reuse, "QSA reuse opt-out");

    // Overrides.
    expect(with("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS", "1").strided_reshape_views == Override::On &&
           with("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS", "0").strided_reshape_views == Override::Off &&
           with("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS", "11").strided_reshape_views == Override::Off, "override values");
    expect(resolve(Override::Unset, true) && !resolve(Override::Unset, false) &&
           resolve(Override::On, false) && !resolve(Override::Off, true), "override resolution");
    expect(with("MOLLM_CUDA_NVFP4_MMA_PREFILL_SORT", "1").nvfp4_prefill_sort == Override::On, "legacy sort name");
    clear();
    setenv("MOLLM_CUDA_NVFP4_PREFILL_SORT", "0", 1);
    setenv("MOLLM_CUDA_NVFP4_MMA_PREFILL_SORT", "1", 1);
    expect(read_runtime_options().nvfp4_prefill_sort == Override::Off, "sort name precedence");

    // Values.
    expect(with("MOLLM_CUDA_Q4_ROW_MAJOR", "1").q4_row_major == 1 &&
           with("MOLLM_CUDA_Q4_ROW_MAJOR", "2").q4_row_major == 2 &&
           with("MOLLM_CUDA_Q4_ROW_MAJOR", "3").q4_row_major == 0, "row-major modes");
    expect(with("MOLLM_CUDA_NVFP4_MMA_GATE_PARTS", "4").nvfp4_mma_gate_parts == 4 &&
           with("MOLLM_CUDA_NVFP4_MMA_GATE_PARTS", "8").nvfp4_mma_gate_parts == 8 &&
           with("MOLLM_CUDA_NVFP4_MMA_GATE_PARTS", "3").nvfp4_mma_gate_parts == 1, "gate parts");
    expect(with("MOLLM_CUDA_PREFILL_FP16_CACHE_MB", "2048").prefill_fp16_cache_bytes == 2048 * mb &&
           with("MOLLM_CUDA_PREFILL_FP16_CACHE_MB", "32769").prefill_fp16_cache_bytes == 0 &&
           with("MOLLM_CUDA_PREFILL_FP16_CACHE_MB", "2g").prefill_fp16_cache_bytes == 0, "FP16 cache budget");
    expect(with("MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB", "512").prefill_gate_scale_cache_bytes == 512 * mb &&
           with("MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB", "8193").prefill_gate_scale_cache_bytes == 0 &&
           with("MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB", " 512").prefill_gate_scale_cache_bytes == 0,
           "gate scale cache budget");
    RuntimeOptions pool = with("MOLLM_CUDA_OUTPUT_POOL_MB", "3072");
    expect(!pool.output_pool_invalid && pool.output_pool_limit_bytes == 3072 * mb, "output pool limit");
    pool = with("MOLLM_CUDA_OUTPUT_POOL_MB", "0");
    expect(!pool.output_pool_invalid && pool.output_pool_limit_bytes == 0, "unlimited output pool");
    expect(with("MOLLM_CUDA_OUTPUT_POOL_MB", "").output_pool_invalid &&
           with("MOLLM_CUDA_OUTPUT_POOL_MB", "-1").output_pool_invalid &&
           with("MOLLM_CUDA_OUTPUT_POOL_MB", "12mb").output_pool_invalid &&
           with("MOLLM_CUDA_OUTPUT_POOL_MB", "99999999999999999999999").output_pool_invalid,
           "invalid output pool limits");

    if (failures) return 1;
    std::printf("CUDA runtime options parse as documented\n");
    return 0;
}
