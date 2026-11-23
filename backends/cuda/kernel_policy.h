#pragma once

// Kernel selection for the CUDA backend, free of CUDA types so the tables can
// be tested on any host. The backend gathers the facts about a request
// (weight layout, shape, alignment) and these functions name the kernel; the
// backend then launches exactly that kernel.

namespace mollm_cuda {

// Which measured specializations a device uses. Every flag defaults to the
// generic path; a flag is set only for an architecture the path was
// qualified on.
struct DevicePolicy {
    // MTP verification: decode-equivalent 2..8-row Q4/GDN arithmetic, Q8 batch
    // argmax, device checkpoints and recurrent state prefixes.
    bool qualified_verification = false;
    int verification_rows_limit = 5; // Wider admission is qualified on SM121.
    bool supports_native_nvfp4_mma = false;
    bool prefer_fp16_expert_palette = false;
    bool prefer_screened_heads = false; // Q4 screen and compact draft heads.
    bool prefer_q4_narrow_row_cache = false; // Exact narrow projection views.
    bool prefer_q4_m3_specializations = false; // Fixed-shape M3 projections.
    bool prefer_q4_fixed_m5 = false; // Compact-wide M5.
    bool prefer_q4_hc_adjacent = false; // Paired HC; also needs its cache configured.
    bool prefer_q4_g128_column_batches = false; // W4G128 M2-M4 one row per warp.
    bool prefer_nvfp4_gate_m1_proven = false; // Resident M1 gate.
    bool prefer_nvfp4_gate_subgroup = false; // Tiny-batch gate.
    bool prefer_nvfp4_mtp_gate_pairs = false; // Exact small-batch gate expert reuse.
    bool prefer_nvfp4_mtp_gate_eight = false; // PAIR gate coalescing.
    bool prefer_nvfp4_m1_down_subwarp = false; // Eight-lane resident M1 down.
    bool prefer_nvfp4_mtp_down_subwarp = false; // Eight-lane small-batch down.
};

// Policy for a device of the given compute capability. The decode
// specializations were qualified on SM121 only; verification batches pair
// with whichever decode kernels a device selects, so they apply from SM80.
inline DevicePolicy make_device_policy(int major, int minor) {
    DevicePolicy policy;
    const bool sm121 = major == 12 && minor == 1;
    policy.qualified_verification = major >= 8;
    policy.verification_rows_limit = sm121 ? 8 : 5;
    policy.supports_native_nvfp4_mma = major == 12 && minor <= 1;
    policy.prefer_fp16_expert_palette = sm121;
    policy.prefer_screened_heads = sm121;
    policy.prefer_q4_narrow_row_cache = sm121;
    policy.prefer_q4_m3_specializations = sm121;
    policy.prefer_q4_fixed_m5 = sm121;
    policy.prefer_q4_hc_adjacent = sm121;
    policy.prefer_q4_g128_column_batches = sm121;
    policy.prefer_nvfp4_gate_m1_proven = sm121;
    policy.prefer_nvfp4_gate_subgroup = sm121;
    policy.prefer_nvfp4_mtp_gate_pairs = sm121;
    policy.prefer_nvfp4_mtp_gate_eight = sm121;
    policy.prefer_nvfp4_m1_down_subwarp = sm121;
    policy.prefer_nvfp4_mtp_down_subwarp = sm121;
    return policy;
}

// A prepared weight as the matmul dispatcher sees it, after its layout
// invariants (group sizes, divisibility of the inner dimension) were checked.
enum class MatmulWeight {
    Other, // Dense FP16/FP32, or a layout only the GEMM path can reject.
    Q8,
    Q4G32,
    Q4G32Compact,
    Q4G32CompactInteger,
    Q4G128,
};

inline bool is_q4_g32(MatmulWeight weight) {
    return weight == MatmulWeight::Q4G32 || weight == MatmulWeight::Q4G32Compact ||
        weight == MatmulWeight::Q4G32CompactInteger;
}

inline bool is_compact_q4_g32(MatmulWeight weight) {
    return weight == MatmulWeight::Q4G32Compact || weight == MatmulWeight::Q4G32CompactInteger;
}

enum class MatmulPath {
    VerifyQ4Compact,  // Exact batch on compact G32 codes.
    VerifyQ4G32,      // Exact batch on canonical G32 tiles; see Q4G32BatchKernel.
    VerifyQ4PerToken, // Row-major sidecar weights: one decode projection per token.
    VerifyQ4G128,     // Exact batch on G128 tiles.
    LegacyQ4G32Batch, // Opt-in approximate small batch.
    Decode,           // Single-token quantized projection; see DecodeKernel.
    Gemm,             // FP16 expansion (or dense weights) through cuBLAS.
};

struct MatmulRequest {
    MatmulWeight weight = MatmulWeight::Other;
    int tokens = 0;
    int columns = 0;
    int inner = 0;
    bool row_major_sidecar = false; // The weight also carries a row-major G32 view.
    bool legacy_small_batch = false; // MOLLM_CUDA_Q4_SMALL_BATCH=1.
};

inline MatmulPath select_matmul_path(const DevicePolicy& policy, const MatmulRequest& request) {
    const bool verification = policy.qualified_verification &&
        request.tokens >= 2 && request.tokens <= 8;
    if (verification && is_q4_g32(request.weight)) {
        if (is_compact_q4_g32(request.weight)) return MatmulPath::VerifyQ4Compact;
        return request.row_major_sidecar ? MatmulPath::VerifyQ4PerToken : MatmulPath::VerifyQ4G32;
    }
    if (verification && request.weight == MatmulWeight::Q4G128) return MatmulPath::VerifyQ4G128;
    if (request.legacy_small_batch && request.weight == MatmulWeight::Q4G32 &&
        request.tokens >= 2 && request.tokens <= 8 && request.inner >= 2048 &&
        static_cast<long long>(request.columns) * request.inner >= 4 * 1024 * 1024)
        return MatmulPath::LegacyQ4G32Batch;
    if (request.weight != MatmulWeight::Other && request.tokens == 1) return MatmulPath::Decode;
    return MatmulPath::Gemm;
}

enum class Q4G32BatchKernel { NarrowRow, M3Adjacent, M3Fixed, Generic };

struct Q4G32BatchRequest {
    int tokens = 0;
    int columns = 0;
    int inner = 0;
    bool narrow_row_eligible = false; // A narrow-row view exists and accepts this batch.
    bool packed_aligned = false; // lda == inner and the activation is float4-aligned.
};

inline Q4G32BatchKernel select_q4_g32_batch(const DevicePolicy& policy,
                                             const Q4G32BatchRequest& request) {
    if (request.narrow_row_eligible) return Q4G32BatchKernel::NarrowRow;
    if (policy.prefer_q4_m3_specializations && request.tokens == 3 && request.packed_aligned) {
        const int n = request.columns, k = request.inner;
        if ((n == 320 && k == 10240) || (n == 640 && k == 2560)) return Q4G32BatchKernel::M3Adjacent;
        if ((n == 2560 && k == 6144) || (n == 512 && k == 2560)) return Q4G32BatchKernel::M3Fixed;
    }
    return Q4G32BatchKernel::Generic;
}

enum class DecodeKernel {
    Q8,
    Q4CompactInteger,
    Q4Compact,
    Q4NarrowRow,
    Q4RowMajor,
    Q4G32,
    Q4G128,
};

struct DecodeRequest {
    MatmulWeight weight = MatmulWeight::Other; // Must not be Other.
    bool narrow_row_eligible = false; // A narrow-row view exists and accepts this token.
    bool row_major_aligned = false; // A row-major view exists and the activation is float2-aligned.
};

inline DecodeKernel select_decode(const DecodeRequest& request) {
    switch (request.weight) {
    case MatmulWeight::Q8: return DecodeKernel::Q8;
    case MatmulWeight::Q4G128: return DecodeKernel::Q4G128;
    case MatmulWeight::Q4G32CompactInteger: return DecodeKernel::Q4CompactInteger;
    case MatmulWeight::Q4G32Compact: return DecodeKernel::Q4Compact;
    default: break;
    }
    if (request.narrow_row_eligible) return DecodeKernel::Q4NarrowRow;
    if (request.row_major_aligned) return DecodeKernel::Q4RowMajor;
    return DecodeKernel::Q4G32;
}

// Rows of a W4G128 tile per warp for an exact verification batch.
inline int q4_g128_batch_rows(const DevicePolicy& policy, int tokens) {
    if (policy.prefer_q4_g128_column_batches && tokens <= 4) return 1;
    return tokens <= 3 ? 2 : 4;
}

// NVFP4 expert projections below the native-MMA row threshold (decode and
// verification batches). Prefill batches at or above it take the MMA paths,
// which moe.cu selects from the prefill batch plan.
enum class Nvfp4GateKernel {
    M1Proven,  // Resident K-major codes, one token, proven layout.
    Paired8,   // K-major codes and scales, pair layout, 2..limit rows.
    Paired,    // K-major codes, 2..5 rows.
    Subgroup,  // K-major codes, tiny batch.
    Code,      // K-major codes, generic; pipelined when the request says so.
    Pipelined, // Canonical codes, pipelined.
    Generic,
};

struct Nvfp4GateRequest {
    int rows = 0;
    int routes = 0;
    int top_k = 0;
    int hidden = 0;
    int inter = 0;
    int verification_rows = 0; // Largest verification batch the backend admits.
    bool codes_kmajor = false;
    bool scales_kmajor = false;
    bool pipeline = false; // The pipelined kernels are enabled for this row count.
    bool resident_mma_layout = false;
    bool pair_layout = false;
    bool m1_layout_proven = false;
    bool activation_aligned = false; // 16-byte aligned input.
};

// A verification batch on K-major gate codes that the paired kernels were
// built for; also what the pair-dispatch counter counts.
inline bool nvfp4_gate_pair_candidate(const DevicePolicy& policy, const Nvfp4GateRequest& request) {
    return policy.prefer_nvfp4_mtp_gate_pairs && request.codes_kmajor &&
        request.rows >= 2 && request.rows <= request.verification_rows &&
        request.top_k == 10 && request.hidden == 2560 && request.inter == 640 && !request.pipeline;
}

inline Nvfp4GateKernel select_nvfp4_gate(const DevicePolicy& policy, const Nvfp4GateRequest& request) {
    if (!request.codes_kmajor)
        return request.pipeline && !request.resident_mma_layout ? Nvfp4GateKernel::Pipelined
                                                                : Nvfp4GateKernel::Generic;
    if (policy.prefer_nvfp4_gate_m1_proven && request.rows == 1 && request.routes == 10 &&
        request.pipeline && request.hidden == 2560 && request.inter == 640 &&
        request.m1_layout_proven && request.activation_aligned)
        return Nvfp4GateKernel::M1Proven;
    const bool paired = nvfp4_gate_pair_candidate(policy, request);
    if (paired && policy.prefer_nvfp4_mtp_gate_eight && request.scales_kmajor && request.pair_layout)
        return Nvfp4GateKernel::Paired8;
    if (paired && request.rows <= 5) return Nvfp4GateKernel::Paired;
    if (policy.prefer_nvfp4_gate_subgroup && request.rows >= 2 &&
        request.rows <= request.verification_rows && request.top_k == 10 && !request.pipeline)
        return Nvfp4GateKernel::Subgroup;
    return Nvfp4GateKernel::Code;
}

enum class Nvfp4DownKernel {
    M1Subwarp,         // Eight-lane exact, one token.
    Pipelined,
    SmallBatchSubwarp, // Eight-lane exact, 2..limit rows.
    Generic,
};

struct Nvfp4DownRequest {
    int rows = 0;
    int top_k = 0;
    int hidden = 0;
    int inter = 0;
    int verification_rows = 0;
    bool pipeline = false;
    bool resident_mma_layout = false;
    bool pair_layout = false;
    bool row_scale_prefetch = false; // The pipelined kernel was asked to prefetch row scales.
};

inline Nvfp4DownKernel select_nvfp4_down(const DevicePolicy& policy, const Nvfp4DownRequest& request) {
    const bool pair_shape = request.top_k == 10 && request.hidden == 2560 && request.inter == 640 &&
        request.pair_layout;
    if (request.pipeline && !request.resident_mma_layout) {
        if (policy.prefer_nvfp4_m1_down_subwarp && request.rows == 1 && pair_shape &&
            !request.row_scale_prefetch)
            return Nvfp4DownKernel::M1Subwarp;
        return Nvfp4DownKernel::Pipelined;
    }
    if (policy.prefer_nvfp4_mtp_down_subwarp && request.rows >= 2 &&
        request.rows <= request.verification_rows && pair_shape && !request.resident_mma_layout)
        return Nvfp4DownKernel::SmallBatchSubwarp;
    return Nvfp4DownKernel::Generic;
}

// Vocabulary and hidden size the screened and compact heads were fitted on;
// every other shape keeps the exact Q8 GEMV.
constexpr int qualified_head_rows = 248320;
constexpr int qualified_head_inner = 2560;

// One hidden state through the language-model head.
enum class ScalarHeadKernel {
    DraftFittedQ4Screen, // Fitted compact Q4 with energy screening (draft confidence).
    DraftCompactQ4,      // Compact Q4 draft head.
    ScreenQ4,            // Exact Q8 argmax behind a Q4 screen.
    ScreenQ6,            // Exact Q8 argmax behind a Q6 screen.
    Matmul,              // The matmul dispatcher's decode projection.
};

struct ScalarHeadRequest {
    int rows = 0;
    int inner = 0;
    bool draft_compact_eligible = false; // A draft request on a Q8 head the compact pack accepts.
    bool probability_output = false; // The draft asked for a confidence.
    bool screen_eligible = false; // A Q8 head, activation and shape the screens accept.
    bool weight_shape_matches = false; // The prepared weight has exactly rows x inner.
};

inline ScalarHeadKernel select_scalar_head(const DevicePolicy& policy, const ScalarHeadRequest& request) {
    const bool qualified_rows = request.rows == qualified_head_rows;
    if (request.draft_compact_eligible)
        return request.probability_output && qualified_rows && request.inner == qualified_head_inner &&
            policy.prefer_screened_heads ? ScalarHeadKernel::DraftFittedQ4Screen
                                         : ScalarHeadKernel::DraftCompactQ4;
    if (request.screen_eligible)
        return policy.prefer_screened_heads && qualified_rows && request.weight_shape_matches
            ? ScalarHeadKernel::ScreenQ4 : ScalarHeadKernel::ScreenQ6;
    return ScalarHeadKernel::Matmul;
}

// Screened Q8 argmax for a verification batch: zero when the batch takes
// the unscreened path, otherwise the widest batch to size the scratch for.
// `candidate` covers the facts the backend checks (layout, shape, alignment,
// no draft request, 3..limit tokens).
inline int q8_head_batch_screen_capacity(const DevicePolicy& policy, int tokens, bool candidate) {
    if (!candidate) return 0;
    if (policy.prefer_screened_heads) return tokens > 5 ? tokens : 5;
    return tokens == 3 ? 3 : 0;
}

// A verification batch through the language-model head.
enum class BatchHeadKernel { ScreenBatch3, ScreenBatch, Q4G128Exact, Q8Exact };

inline BatchHeadKernel select_batch_head(const DevicePolicy& policy, int tokens, bool screen_candidate,
                                         bool q4_g128) {
    if (q8_head_batch_screen_capacity(policy, tokens, screen_candidate) > 0)
        return tokens == 3 ? BatchHeadKernel::ScreenBatch3 : BatchHeadKernel::ScreenBatch;
    return q4_g128 ? BatchHeadKernel::Q4G128Exact : BatchHeadKernel::Q8Exact;
}

// The paired HC projections of one decode token.
enum class HcPairKernel { CachedCopy, Adjacent, Unequal };

inline HcPairKernel select_hc_pair(const DevicePolicy& policy, bool second_of_pair, bool input_aligned) {
    if (second_of_pair) return HcPairKernel::CachedCopy;
    return policy.prefer_q4_hc_adjacent && input_aligned ? HcPairKernel::Adjacent : HcPairKernel::Unequal;
}

inline const char* weight_name(MatmulWeight weight) {
    switch (weight) {
    case MatmulWeight::Other: return "dense";
    case MatmulWeight::Q8: return "q8";
    case MatmulWeight::Q4G32: return "q4_g32";
    case MatmulWeight::Q4G32Compact: return "q4_g32_compact";
    case MatmulWeight::Q4G32CompactInteger: return "q4_g32_compact_integer";
    case MatmulWeight::Q4G128: return "q4_g128";
    }
    return "unknown";
}

inline const char* kernel_name(MatmulPath path) {
    switch (path) {
    case MatmulPath::VerifyQ4Compact: return "q4_compact_small_batch_exact";
    case MatmulPath::VerifyQ4G32: return "q4_g32_small_batch_exact";
    case MatmulPath::VerifyQ4PerToken: return "q4_per_token_decode";
    case MatmulPath::VerifyQ4G128: return "q4_g128_small_batch_exact";
    case MatmulPath::LegacyQ4G32Batch: return "q4_g32_small_batch";
    case MatmulPath::Decode: return "decode";
    case MatmulPath::Gemm: return "gemm";
    }
    return "unknown";
}

inline const char* kernel_name(Q4G32BatchKernel kernel) {
    switch (kernel) {
    case Q4G32BatchKernel::NarrowRow: return "q4_narrow_row";
    case Q4G32BatchKernel::M3Adjacent: return "q4_g32_m3_adjacent";
    case Q4G32BatchKernel::M3Fixed: return "q4_g32_m3_fixed";
    case Q4G32BatchKernel::Generic: return "q4_g32_small_batch_exact";
    }
    return "unknown";
}

inline const char* kernel_name(DecodeKernel kernel) {
    switch (kernel) {
    case DecodeKernel::Q8: return "q8_dense_gemv";
    case DecodeKernel::Q4CompactInteger: return "q4_compact_integer_gemv";
    case DecodeKernel::Q4Compact: return "q4_g32_compact_gemv";
    case DecodeKernel::Q4NarrowRow: return "q4_narrow_row";
    case DecodeKernel::Q4RowMajor: return "q4_row_g32_gemv";
    case DecodeKernel::Q4G32: return "q4_g32_dense_gemv";
    case DecodeKernel::Q4G128: return "q4_g128_dense_gemv";
    }
    return "unknown";
}

inline const char* kernel_name(Nvfp4GateKernel kernel) {
    switch (kernel) {
    case Nvfp4GateKernel::M1Proven: return "nvfp4_gate_code_m1_proven";
    case Nvfp4GateKernel::Paired8: return "nvfp4_gate_paired8";
    case Nvfp4GateKernel::Paired: return "nvfp4_gate_paired";
    case Nvfp4GateKernel::Subgroup: return "nvfp4_gate_code_subgroup";
    case Nvfp4GateKernel::Code: return "nvfp4_gate_code";
    case Nvfp4GateKernel::Pipelined: return "nvfp4_pipelined";
    case Nvfp4GateKernel::Generic: return "nvfp4_generic";
    }
    return "unknown";
}

inline const char* kernel_name(Nvfp4DownKernel kernel) {
    switch (kernel) {
    case Nvfp4DownKernel::M1Subwarp: return "nvfp4_exact_subwarp_m1";
    case Nvfp4DownKernel::Pipelined: return "nvfp4_pipelined";
    case Nvfp4DownKernel::SmallBatchSubwarp: return "nvfp4_exact_subwarp_small_batch";
    case Nvfp4DownKernel::Generic: return "nvfp4_generic";
    }
    return "unknown";
}

inline const char* kernel_name(ScalarHeadKernel kernel) {
    switch (kernel) {
    case ScalarHeadKernel::DraftFittedQ4Screen: return "draft_fitted_q4_screen";
    case ScalarHeadKernel::DraftCompactQ4: return "draft_compact_q4";
    case ScalarHeadKernel::ScreenQ4: return "q8_head_screen_q4";
    case ScalarHeadKernel::ScreenQ6: return "q8_head_screen_q6";
    case ScalarHeadKernel::Matmul: return "matmul";
    }
    return "unknown";
}

inline const char* kernel_name(BatchHeadKernel kernel) {
    switch (kernel) {
    case BatchHeadKernel::ScreenBatch3: return "q8_head_screen_batch3";
    case BatchHeadKernel::ScreenBatch: return "q8_head_screen_batch";
    case BatchHeadKernel::Q4G128Exact: return "q4_g128_small_batch_exact";
    case BatchHeadKernel::Q8Exact: return "q8_small_batch_exact";
    }
    return "unknown";
}

inline const char* kernel_name(HcPairKernel kernel) {
    switch (kernel) {
    case HcPairKernel::CachedCopy: return "hc_cached_copy";
    case HcPairKernel::Adjacent: return "q4_g32_hc_adjacent";
    case HcPairKernel::Unequal: return "q4_g32_unequal";
    }
    return "unknown";
}

}  // namespace mollm_cuda
