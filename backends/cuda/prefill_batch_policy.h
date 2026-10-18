#pragma once

namespace mollm_cuda {
// Qualified logical-prefill defaults; explicit overrides retain the existing
// exact "1" opt-in semantics, including "0" disabling the optimization.
constexpr bool logical_prefill_option(const char* option,bool logical_batch) {
    return option ? option[0]=='1' && option[1]=='\0' : logical_batch;
}
// Logical batching and paired-window aggregation share operator eligibility,
// not scheduling, tensor ownership, or concat/split allocations.
struct Nvfp4PrefillBatchPolicy {
    bool logical_batch = false;
    bool merged_windows = false;
    constexpr bool large_batch() const { return logical_batch || merged_windows; }
    constexpr bool fused_rows(int rows) const {
        return rows >= 16 && rows <= (large_batch() ? 2048 : 1024);
    }
    constexpr bool kmajor_rows(int rows) const {
        return rows<=2048 && ((logical_batch&&rows>=16)||(merged_windows&&rows>1024));
    }
};
}
