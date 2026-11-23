#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mollm_cuda {
struct Nvfp4Projection;
struct Nvfp4MmaBatch;

// One routed expert projection. Pointers may refer to coherent system memory
// only after the backend has checked pageable-memory/host-page-table support.
struct Nvfp4Projection {
    const uint8_t* weight;
    // FP8 block scales. FP16 palette jobs (launch_fp16_palette_projections)
    // store their __half palette here instead, keeping the descriptor ABI.
    const uint8_t* block_scales;
    const float* row_scales;
    const float* input;
    float* output;
    bool pair_layout;
    bool mma_layout = false;
    // Resident gate-only [K/64][1280][4] SF bytes; not a package flag.
    bool gate_scales_kmajor = false;
    float input_global = 0;
    float input_global_second = 0;
    float output_scale = 1;
};
static_assert(sizeof(Nvfp4Projection)==56,"resident descriptor ABI changed");

// Same fixed transpose as gate_scale_view, for one expert's bounded load-time
// staging. 204800 bytes, 4-byte aligned, disjoint; no allocation or sync.
cudaError_t launch_repack_nvfp4_gate_scales(const Nvfp4Projection& source,
    uint8_t* target, int rows, int inner, cudaStream_t stream = nullptr);

// Fixed 1280x2560 gate/up: byte-exact [1280][40][4] -> [40][1280][4].
// Source/target tables contain 2*experts descriptors (gate, then down).
// Caller owns device-readable, 4-byte-aligned indirect scale extents and
// disjoint source tables, target tables and experts*204800-byte scale storage.
// Clones all descriptors; only target gate block_scales pointers change.
cudaError_t launch_nvfp4_gate_scale_view(const Nvfp4Projection* source,
    Nvfp4Projection* target, uint8_t* scales, int experts,
    cudaStream_t stream = nullptr);

// Canonical row/pair layouts: preserve original 32-lane FMA chains/tree while
// mapping each row to 16/8 lanes. Backend policy selects qualified shapes.
cudaError_t launch_nvfp4_exact_subwarp(const Nvfp4Projection* jobs,int count,
    int rows,int inner,int lanes,cudaStream_t stream = nullptr);
// Prepare one projection in the native m16n8k64 register/scale layout.
// Targets hold ceil(rows/16)*16*K/2 weight bytes and /16 scale bytes.
// Source/target ranges must not overlap. Row-global scales are unchanged.
// Does not allocate or synchronize; callers retain source staging until the
// stream completes. This permits bounded staging into one resident payload.
cudaError_t launch_repack_nvfp4_mma(const Nvfp4Projection& source,
    uint8_t* target_weights, uint8_t* target_scales, int rows, int inner,
    cudaStream_t stream = nullptr);

// Fixed 1280x2560 gate code transpose: canonical ROW/PAIR -> K64-major.
// Source and target each hold exactly 1638400 code bytes, are 16-byte aligned
// and disjoint. Scales and descriptors are not modified. No allocation/sync;
// retain the source staging until stream completion. Not a runtime selector.
cudaError_t launch_repack_nvfp4_gate_codes(const Nvfp4Projection& source,
    uint8_t* target_weights, int rows, int inner,
    cudaStream_t stream = nullptr);
// Fixed gate-code view consumer; row scales canonical, block scales selected
// by the per-descriptor gate_scales_kmajor flag.
// All jobs use the K64-major view above, never native MMA/interleaved storage.
// Same ordered32-lane FP32 chains/reduction as canonical SIMT projections.
cudaError_t launch_nvfp4_gate_code_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, bool pipeline_inputs = true,
    cudaStream_t stream = nullptr);

// Resident M1 gate, N1280/K2560 and 1..50 jobs, without per-descriptor checks.
// PRECONDITION: every descriptor is PAIR, SF K-major and non-MMA, with 16-byte
// aligned code/input pointers and 4-byte aligned block scales. The kernel does
// not verify this; any other layout produces wrong results silently. Callers
// that cannot guarantee it must use launch_nvfp4_gate_code_projections.
// FP32 FMA chains, reduction and row scaling match that general entry point.
cudaError_t launch_nvfp4_gate_code_m1_proven_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);

// Same fixed K64-major gate view and exact 32-lane arithmetic as above,
// scheduled as two independent chains in each physical 16-lane row.
cudaError_t launch_nvfp4_gate_code_subgroup_projections(const Nvfp4Projection* jobs,
    int count,int rows,int inner,cudaStream_t stream = nullptr);
// Exact canonical ROW/PAIR gate codes in K64-major order, N1280/K2560;
// row-major or K-major scales, FP32 input/output, never MMA layout.
// Pair equal experts among 1..50 jobs without scratch or descriptor reordering.
cudaError_t launch_nvfp4_gate_paired_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);
// Same exact canonical K64-major gate arithmetic; PAIR rows use eight lanes.
// ROW descriptors keep the 16-lane tree inside each 16-row tile.
cudaError_t launch_nvfp4_gate_paired8_projections(const Nvfp4Projection* jobs,
    int count,int rows,int inner,cudaStream_t stream = nullptr);

void launch_nvfp4_projections(const Nvfp4Projection* jobs, int count,
                              int rows, int inner, cudaStream_t stream = nullptr,
                              bool mma_layout = false);
// Preserves FP32 accumulation/reduction order. No allocation or synchronization.
// One-step packet pipeline. prefetch_row_scale adds an L2 prefetch of the
// row scale on entry.
cudaError_t launch_nvfp4_pipelined_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream = nullptr, bool prefetch_row_scale = false);
// Reuse the routed descriptor ABI for FP16 expert-major matrices. Only
// weight/input/output fields are read; route weights are applied by activation.
cudaError_t launch_fp16_expert_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);
// Same descriptor ABI for W4G128 expert-major matrices (Q4B8G128Block rows,
// rows % 8 == 0, inner % 128 == 0). Each route uses decode GEMV arithmetic.
cudaError_t launch_q4_g128_expert_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);
// Experimental fixed-checkpoint activation quantization. Native packed
// descriptors require positive input globals; a split selects the second
// global for up rows. canonical_weights reads row/pair weights and row-major
// block scales directly; otherwise native fragment-plane layout is required.
// No allocation or synchronization. All jobs must use the selected layout.
cudaError_t launch_nvfp4_mma_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int split_row = 0, cudaStream_t stream = nullptr,
    float* partial = nullptr, int parts = 1, bool canonical_weights = false,
    uint8_t* packed_inputs = nullptr, uint8_t* input_scales = nullptr,
    bool gate_codes_kmajor = false); // Already prepared input requires split_row==0.
struct Nvfp4MmaBatch { int first; int size; };
// Fused prefill path for one grouped batch plan: gate/up native-FP4 MMA,
// SiLU-gate activation in BF16, and requantization of the activations into the
// down projection's packed inputs and scales. All buffers must be disjoint.
// Jobs must keep equal-expert descriptors adjacent; batches holds count entries
// and is rebuilt on every call. Arguments are checked before anything is enqueued;
// the caller validates descriptor geometry, pointer extents and lifetimes.
// Each layout option requires the ones before it: compact_logits needs both
// scale views, token_input needs compact_logits, pair_words needs token_input,
// weight_codes_kmajor needs pair_words, token_codes_kmajor needs both.
struct Nvfp4FusedActivationOptions {
    bool publish_activated = false;   // Also write the FP32 activations.
    bool transposed_scales = false;   // Gate block scales in the K-major view.
    bool kmajor_input_scales = false; // Quantized input scales stored K-major.
    bool compact_logits = false;      // BF16 logits kept as 16-bit codes.
    // Quantize each prompt token once instead of once per route. The caller
    // guarantees every gate/up global equals token_global and every descriptor
    // input is a complete row of token_input[token_count][2560]. token_indices
    // has count writable ints and is rebuilt on every call.
    const float* token_input = nullptr;
    int token_count = 0;
    float token_global = 1.f;
    int* token_indices = nullptr;
    bool pair_words = false;          // Token codes loaded as 8-byte pairs.
    bool weight_codes_kmajor = false; // Resident K-major gate codes.
    bool token_codes_kmajor = false;  // Token codes stored K-major.
};
cudaError_t launch_nvfp4_mma_fused_activation(const Nvfp4Projection* jobs,
    const Nvfp4Projection* down_jobs,int count,Nvfp4MmaBatch* batches,
    uint8_t* packed_inputs,uint8_t* input_scales,float* activated,
    uint8_t* down_packed,uint8_t* down_scales,
    const Nvfp4FusedActivationOptions& options = {},cudaStream_t stream = nullptr);
// Diagnostic only, both choices include identical sentinel initialization.
cudaError_t launch_nvfp4_batch_plan_diagnostic(const Nvfp4Projection* jobs,int count,
    Nvfp4MmaBatch* batches,int columns,bool parallel,cudaStream_t stream=nullptr);
// Native-FP4 MMA for a prefill batch: groups jobs that share an expert into
// batches of up to max_batch_size columns, quantizes their inputs and runs one
// MMA per batch. batches holds count entries and is rebuilt on every call.
struct Nvfp4GroupedOptions {
    int split_row = 0;                 // Rows of the gate half when gate and up are stacked; 0 for one matrix.
    uint8_t* packed_inputs = nullptr;  // Scratch for quantized inputs, or the inputs when inputs_prepared.
    uint8_t* input_scales = nullptr;   // Scratch for their block scales.
    int min_batch_size = 1;            // Smaller groups take the per-job path.
    bool canonical_weights = false;    // Weights in the package layout rather than native fragment order.
    bool prepare_singles = false;
    Nvfp4MmaBatch* compact_batches = nullptr; // Scratch for a dense batch list; needs compact_grid_y.
    int compact_grid_y = 0;
    int max_batch_size = 8;
    bool specialized_prepared = false;
    bool inputs_prepared = false;      // packed_inputs and input_scales already hold this batch's inputs.
    bool async_weights = false;        // Stage weights with asynchronous copies.
    bool output_bf16 = false;          // Outputs are tightly packed BF16 rows.
    bool early_inputs = false;
    bool parallel_plan = false;        // Build the batch plan in parallel; excludes the compact list.
    bool vector_quantize = false;      // Gate 1280x2560 only, with output_bf16 and async_weights.
    bool gate_codes_kmajor = false;    // Resident K-major gate codes; canonical 1280x2560 only.
};
cudaError_t launch_nvfp4_mma_grouped(const Nvfp4Projection* jobs, int count,
    Nvfp4MmaBatch* batches, int rows, int inner,
    const Nvfp4GroupedOptions& options = {}, cudaStream_t stream = nullptr);
// Ungrouped routing, up to 1024 experts / 32 selected experts. The immutable
// tables contain gate then down descriptors in expert-ID order; only input /
// output fields of the selected descriptors are filled for this invocation.
void launch_moe_resident_routes(const float* logits, int rows, int experts,
    int top_k, int score_func, bool normalize, float scale,
    const Nvfp4Projection* tables, const float* input, float* gate_up,
    float* activated, float* down, int hidden, int inter,
    Nvfp4Projection* gate_jobs, Nvfp4Projection* down_jobs, float* weights,
    cudaStream_t stream = nullptr, int cached_best_override = -1,
    bool down_output_bf16 = false, bool gate_output_bf16 = false);
// Override is diagnostic: -1 normal register selector, 0 original, 1 cached maxima.
// down_output_bf16 changes only output pointers to tightly packed uint16 rows;
// caller must pair with the BF16 grouped producer and combine, never FLOAT paths.
// gate_output_bf16 analogously packs gate/up pointers; requires the BF16 async
// gate producer and activation preparation consumer as an inseparable pair.
// Group a small prefill's jobs by immutable expert weight address. Output
// pointers retain the original token/route slots, so activation and reduction
// order do not change. Default: at most4096 jobs (32 KiB shared keys).
// Diagnostic tiled_keys uses fixed1 KiB shared storage, up to65535 jobs;
// it retains quadratic comparison work and is not a throughput claim.
void launch_group_nvfp4_jobs(const Nvfp4Projection* gate_jobs,
    const Nvfp4Projection* down_jobs, Nvfp4Projection* grouped_gate,
    Nvfp4Projection* grouped_down, int count, cudaStream_t stream = nullptr,
    Nvfp4MmaBatch* pair_ranks = nullptr, bool tiled_keys = false);
// Stable radix grouping by the full unsigned weight address, preserving the
// original order of equal keys and both descriptors' route/output pointers.
// count is in [0,65535]. No allocation or synchronization; capture safe when
// caller-provided scratch is sized in advance. Scratch must be 256-byte aligned
// and disjoint from descriptor ranges. Output ranges must not overlap each
// other or either input range; the two read-only input ranges may overlap.
// This API does not produce the optional legacy SIMT pair plan.
cudaError_t nvfp4_radix_group_scratch_size(int count, size_t* bytes);
cudaError_t launch_group_nvfp4_jobs_radix(const Nvfp4Projection* gate_jobs,
    const Nvfp4Projection* down_jobs, Nvfp4Projection* grouped_gate,
    Nvfp4Projection* grouped_down, int count, void* scratch, size_t scratch_bytes,
    cudaStream_t stream = nullptr);
void launch_moe_activation(const float* gate, const float* up,
                            const float* route_weights, float* output,
                            int rows, int width, int stride, cudaStream_t stream = nullptr);
// Diagnostic activation + quantization isolation. Unweighted routed activation;
// jobs' inputs must uniquely cover the count rows of activated, in any order.
// Packed/scales are job-order, activated remains original route-order.
// fusion_mode: 0 original two launches, 1 scalar fusion, 2 subgroup fusion.
// Mode3: subgroup packed-only; activated is a pointer base and is NOT written.
// Caller must ensure every downstream projection uses prepared packed inputs.
cudaError_t launch_nvfp4_activation_prepare(const Nvfp4Projection* jobs,
    int count, int inner, const float* gate_up, float* activated,
    uint8_t* packed, uint8_t* scales, int fusion_mode, cudaStream_t stream = nullptr,
    bool gate_input_bf16 = false);
// gate_input_bf16: inner640/subgroup modes2,3 only; gate_up references tightly
// packed uint16 gate/up codes using the same existing BF16 input boundary.
void launch_moe_combine(const float* routed, const float* shared,
                         const float* shared_gate, float* output,
                         int rows, int hidden, int top_k, cudaStream_t stream = nullptr);
cudaError_t launch_moe_combine_bf16(const uint16_t* routed,const float* shared,
    const float* shared_gate,float* output,int rows,int hidden,int top_k,
    cudaStream_t stream=nullptr);

}  // namespace mollm_cuda
