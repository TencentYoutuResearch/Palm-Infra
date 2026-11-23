// The CUDA kernel selection tables, checked without a CUDA toolkit.
#include "backends/cuda/kernel_policy.h"

#include <cstdio>
#include <initializer_list>

using namespace mollm_cuda;

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

MatmulPath path(const DevicePolicy& policy, MatmulWeight weight, int tokens, int columns = 2560,
                int inner = 2560, bool sidecar = false, bool legacy = false) {
    MatmulRequest request;
    request.weight = weight;
    request.tokens = tokens;
    request.columns = columns;
    request.inner = inner;
    request.row_major_sidecar = sidecar;
    request.legacy_small_batch = legacy;
    return select_matmul_path(policy, request);
}

Q4G32BatchKernel batch(const DevicePolicy& policy, int tokens, int columns, int inner,
                       bool narrow = false, bool packed = true) {
    Q4G32BatchRequest request;
    request.tokens = tokens;
    request.columns = columns;
    request.inner = inner;
    request.narrow_row_eligible = narrow;
    request.packed_aligned = packed;
    return select_q4_g32_batch(policy, request);
}

DecodeKernel decode(MatmulWeight weight, bool narrow = false, bool row_major = false) {
    DecodeRequest request;
    request.weight = weight;
    request.narrow_row_eligible = narrow;
    request.row_major_aligned = row_major;
    return select_decode(request);
}

}  // namespace

int main() {
    const DevicePolicy sm121 = make_device_policy(12, 1);
    const DevicePolicy sm120 = make_device_policy(12, 0);
    const DevicePolicy sm89 = make_device_policy(8, 9);
    const DevicePolicy sm75 = make_device_policy(7, 5);

    // Device policy: verification from SM80, specializations on SM121 only.
    expect(sm121.qualified_verification && sm120.qualified_verification &&
           sm89.qualified_verification && !sm75.qualified_verification,
           "qualified verification starts at SM80");
    expect(sm121.verification_rows_limit == 8 && sm89.verification_rows_limit == 5 &&
           sm120.verification_rows_limit == 5, "eight verification rows only on SM121");
    expect(sm121.supports_native_nvfp4_mma && sm120.supports_native_nvfp4_mma &&
           !sm89.supports_native_nvfp4_mma && !make_device_policy(12, 2).supports_native_nvfp4_mma,
           "native NVFP4 MMA on SM120 and SM121");
    for (const DevicePolicy* generic : {&sm120, &sm89, &sm75})
        expect(!generic->prefer_fp16_expert_palette && !generic->prefer_screened_heads &&
               !generic->prefer_q4_narrow_row_cache && !generic->prefer_q4_m3_specializations &&
               !generic->prefer_q4_fixed_m5 && !generic->prefer_q4_hc_adjacent &&
               !generic->prefer_q4_g128_column_batches && !generic->prefer_nvfp4_gate_m1_proven &&
               !generic->prefer_nvfp4_gate_subgroup && !generic->prefer_nvfp4_mtp_gate_pairs &&
               !generic->prefer_nvfp4_mtp_gate_eight && !generic->prefer_nvfp4_m1_down_subwarp &&
               !generic->prefer_nvfp4_mtp_down_subwarp,
               "devices other than SM121 keep the generic paths");
    expect(sm121.prefer_fp16_expert_palette && sm121.prefer_screened_heads &&
           sm121.prefer_q4_narrow_row_cache && sm121.prefer_q4_m3_specializations &&
           sm121.prefer_q4_fixed_m5 && sm121.prefer_q4_hc_adjacent &&
           sm121.prefer_q4_g128_column_batches && sm121.prefer_nvfp4_gate_m1_proven &&
           sm121.prefer_nvfp4_gate_subgroup && sm121.prefer_nvfp4_mtp_gate_pairs &&
           sm121.prefer_nvfp4_mtp_gate_eight && sm121.prefer_nvfp4_m1_down_subwarp &&
           sm121.prefer_nvfp4_mtp_down_subwarp, "SM121 enables its specializations");

    // Matmul path: verification batches of 2..8 tokens stay exact per layout.
    for (int tokens = 2; tokens <= 8; ++tokens) {
        expect(path(sm89, MatmulWeight::Q4G32, tokens) == MatmulPath::VerifyQ4G32, "G32 batch");
        expect(path(sm89, MatmulWeight::Q4G32, tokens, 2560, 2560, true) ==
               MatmulPath::VerifyQ4PerToken, "row-major sidecar runs per token");
        expect(path(sm89, MatmulWeight::Q4G32Compact, tokens, 2560, 2560, true) ==
               MatmulPath::VerifyQ4Compact, "compact batch ignores the sidecar");
        expect(path(sm89, MatmulWeight::Q4G32CompactInteger, tokens) ==
               MatmulPath::VerifyQ4Compact, "integer compact batch");
        expect(path(sm89, MatmulWeight::Q4G128, tokens) == MatmulPath::VerifyQ4G128, "G128 batch");
        expect(path(sm89, MatmulWeight::Q8, tokens) == MatmulPath::Gemm, "Q8 batch expands");
        expect(path(sm89, MatmulWeight::Other, tokens) == MatmulPath::Gemm, "dense batch");
        expect(path(sm75, MatmulWeight::Q4G32, tokens) == MatmulPath::Gemm &&
               path(sm75, MatmulWeight::Q4G128, tokens) == MatmulPath::Gemm,
               "no exact batches without qualified verification");
    }
    expect(path(sm121, MatmulWeight::Q4G32, 9) == MatmulPath::Gemm &&
           path(sm121, MatmulWeight::Q4G128, 9) == MatmulPath::Gemm, "nine tokens expand");
    for (MatmulWeight weight : {MatmulWeight::Q8, MatmulWeight::Q4G32, MatmulWeight::Q4G32Compact,
                                MatmulWeight::Q4G32CompactInteger, MatmulWeight::Q4G128})
        expect(path(sm75, weight, 1) == MatmulPath::Decode && path(sm121, weight, 1) == MatmulPath::Decode,
               "one quantized token decodes");
    expect(path(sm121, MatmulWeight::Other, 1) == MatmulPath::Gemm, "one dense token uses GEMM");
    // The opt-in approximate batch only applies where verification does not.
    expect(path(sm75, MatmulWeight::Q4G32, 4, 2560, 2560, false, true) == MatmulPath::LegacyQ4G32Batch,
           "legacy batch on a large canonical projection");
    expect(path(sm75, MatmulWeight::Q4G32, 4, 512, 2560, false, true) == MatmulPath::Gemm,
           "legacy batch needs four million weights");
    expect(path(sm75, MatmulWeight::Q4G32, 4, 10240, 320, false, true) == MatmulPath::Gemm,
           "legacy batch needs a wide inner dimension");
    expect(path(sm75, MatmulWeight::Q4G32Compact, 4, 2560, 2560, false, true) == MatmulPath::Gemm,
           "legacy batch skips compact weights");
    expect(path(sm89, MatmulWeight::Q4G32, 4, 2560, 2560, false, true) == MatmulPath::VerifyQ4G32,
           "verification outranks the legacy batch");

    // G32 batch kernels.
    expect(batch(sm121, 3, 320, 10240) == Q4G32BatchKernel::M3Adjacent &&
           batch(sm121, 3, 640, 2560) == Q4G32BatchKernel::M3Adjacent, "M3 adjacent shapes");
    expect(batch(sm121, 3, 2560, 6144) == Q4G32BatchKernel::M3Fixed &&
           batch(sm121, 3, 512, 2560) == Q4G32BatchKernel::M3Fixed, "M3 fixed shapes");
    expect(batch(sm121, 3, 1024, 2560) == Q4G32BatchKernel::Generic, "other M3 shapes");
    expect(batch(sm121, 4, 640, 2560) == Q4G32BatchKernel::Generic, "M3 kernels need three tokens");
    expect(batch(sm121, 3, 640, 2560, false, false) == Q4G32BatchKernel::Generic,
           "M3 kernels need packed aligned activations");
    expect(batch(sm89, 3, 640, 2560) == Q4G32BatchKernel::Generic &&
           batch(sm89, 3, 2560, 6144) == Q4G32BatchKernel::Generic, "M3 kernels are SM121 only");
    expect(batch(sm121, 3, 2560, 6144, true) == Q4G32BatchKernel::NarrowRow &&
           batch(sm89, 4, 2560, 6144, true) == Q4G32BatchKernel::NarrowRow,
           "an eligible narrow-row view wins");

    // Single-token kernels.
    expect(decode(MatmulWeight::Q8) == DecodeKernel::Q8 &&
           decode(MatmulWeight::Q4G128) == DecodeKernel::Q4G128 &&
           decode(MatmulWeight::Q4G32CompactInteger, true, true) == DecodeKernel::Q4CompactInteger &&
           decode(MatmulWeight::Q4G32Compact, true, true) == DecodeKernel::Q4Compact,
           "layout-determined decode kernels");
    expect(decode(MatmulWeight::Q4G32, true, true) == DecodeKernel::Q4NarrowRow &&
           decode(MatmulWeight::Q4G32, false, true) == DecodeKernel::Q4RowMajor &&
           decode(MatmulWeight::Q4G32) == DecodeKernel::Q4G32, "canonical G32 decode order");

    // W4G128 verification rows per warp.
    for (int tokens = 2; tokens <= 8; ++tokens) {
        expect(q4_g128_batch_rows(sm89, tokens) == (tokens <= 3 ? 2 : 4), "generic G128 rows");
        expect(q4_g128_batch_rows(sm121, tokens) == (tokens <= 4 ? 1 : 4), "SM121 G128 rows");
    }

    // NVFP4 expert gate below the native-MMA threshold.
    {
        Nvfp4GateRequest decode_gate;
        decode_gate.rows = 1;
        decode_gate.routes = 10;
        decode_gate.top_k = 10;
        decode_gate.hidden = 2560;
        decode_gate.inter = 640;
        decode_gate.verification_rows = 8;
        decode_gate.codes_kmajor = true;
        decode_gate.scales_kmajor = true;
        decode_gate.pipeline = true;
        decode_gate.pair_layout = true;
        decode_gate.m1_layout_proven = true;
        decode_gate.activation_aligned = true;
        expect(select_nvfp4_gate(sm121, decode_gate) == Nvfp4GateKernel::M1Proven, "SM121 proven M1 gate");
        expect(select_nvfp4_gate(sm89, decode_gate) == Nvfp4GateKernel::Code, "generic M1 gate");
        Nvfp4GateRequest unaligned = decode_gate;
        unaligned.activation_aligned = false;
        expect(select_nvfp4_gate(sm121, unaligned) == Nvfp4GateKernel::Code, "unaligned M1 gate");
        Nvfp4GateRequest unproven = decode_gate;
        unproven.m1_layout_proven = false;
        expect(select_nvfp4_gate(sm121, unproven) == Nvfp4GateKernel::Code, "unproven M1 gate");

        // Verification batches run without the pipelined kernels.
        Nvfp4GateRequest verify = decode_gate;
        verify.pipeline = false;
        for (int rows = 2; rows <= 8; ++rows) {
            verify.rows = rows;
            verify.routes = rows * 10;
            Nvfp4GateRequest request = verify;
            expect(select_nvfp4_gate(sm121, request) == Nvfp4GateKernel::Paired8, "SM121 paired-eight gate");
            expect(nvfp4_gate_pair_candidate(sm121, request), "SM121 pair candidate");
            expect(select_nvfp4_gate(sm89, request) == Nvfp4GateKernel::Code, "generic batch gate");
            expect(!nvfp4_gate_pair_candidate(sm89, request), "generic pair candidate");
            request.pair_layout = false;
            expect(select_nvfp4_gate(sm121, request) ==
                   (rows <= 5 ? Nvfp4GateKernel::Paired : Nvfp4GateKernel::Subgroup),
                   "SM121 gate without the pair layout");
            expect(nvfp4_gate_pair_candidate(sm121, request), "pair candidate without the pair layout");
            request.hidden = 2048;
            expect(select_nvfp4_gate(sm121, request) == Nvfp4GateKernel::Subgroup, "SM121 subgroup gate");
            request.top_k = 8;
            expect(select_nvfp4_gate(sm121, request) == Nvfp4GateKernel::Code, "gate with another top-k");
            request = verify;
            request.pipeline = true;
            expect(select_nvfp4_gate(sm121, request) == Nvfp4GateKernel::Code, "pipelined batch gate");
        }
        verify.rows = 6;
        verify.verification_rows = 5;
        expect(select_nvfp4_gate(sm121, verify) == Nvfp4GateKernel::Code, "gate beyond the verification limit");

        Nvfp4GateRequest canonical = decode_gate;
        canonical.codes_kmajor = false;
        canonical.scales_kmajor = false;
        expect(select_nvfp4_gate(sm121, canonical) == Nvfp4GateKernel::Pipelined, "canonical pipelined gate");
        canonical.resident_mma_layout = true;
        expect(select_nvfp4_gate(sm121, canonical) == Nvfp4GateKernel::Generic, "MMA-layout gate");
        canonical.resident_mma_layout = false;
        canonical.pipeline = false;
        expect(select_nvfp4_gate(sm121, canonical) == Nvfp4GateKernel::Generic, "canonical gate without pipeline");
    }

    // NVFP4 expert down projection below the native-MMA threshold.
    {
        Nvfp4DownRequest down;
        down.rows = 1;
        down.top_k = 10;
        down.hidden = 2560;
        down.inter = 640;
        down.verification_rows = 8;
        down.pipeline = true;
        down.pair_layout = true;
        expect(select_nvfp4_down(sm121, down) == Nvfp4DownKernel::M1Subwarp, "SM121 M1 down");
        expect(select_nvfp4_down(sm89, down) == Nvfp4DownKernel::Pipelined, "generic M1 down");
        Nvfp4DownRequest request = down;
        request.row_scale_prefetch = true;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::Pipelined, "M1 down with prefetch");
        request = down;
        request.pair_layout = false;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::Pipelined, "M1 down without the pair layout");
        request = down;
        request.rows = 3;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::Pipelined, "pipelined batch down");
        request.pipeline = false;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::SmallBatchSubwarp, "SM121 batch down");
        expect(select_nvfp4_down(sm89, request) == Nvfp4DownKernel::Generic, "generic batch down");
        request.rows = 9;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::Generic, "down beyond the verification limit");
        request.rows = 3;
        request.resident_mma_layout = true;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::Generic, "MMA-layout down");
        request = down;
        request.resident_mma_layout = true;
        expect(select_nvfp4_down(sm121, request) == Nvfp4DownKernel::Generic, "MMA-layout M1 down");
    }

    if (failures) return 1;
    std::printf("CUDA kernel selection tables match\n");
    return 0;
}
