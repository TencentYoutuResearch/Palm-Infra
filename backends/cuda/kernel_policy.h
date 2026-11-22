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

}  // namespace mollm_cuda
