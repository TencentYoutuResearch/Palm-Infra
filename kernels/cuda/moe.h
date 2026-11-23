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
cudaError_t launch_nvfp4_pipelined_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, bool pipeline_inputs = true, cudaStream_t stream = nullptr,
    bool cache_input = false, int load_hint = 0, bool direct_fp4 = false,
    bool fixed_inner = false, bool warp_packet = false,
    bool aligned_packets = false, bool direct_scale = false, bool route_major = false,
    bool pair_only = false, bool broadcast_scale = false); // Broadcast requires K%128==0. Pair-only caller guarantees canonical pair layout; aligned caller guarantees 2-byte weights / 16-byte inputs.
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
// Jobs must have adjacent equal-expert descriptors (weight and scale pointers).
// Scratch holds count batches. GPU-only grouping; stream ordered and graph safe.
// Diagnostic fixed gate1280/K2560 followed by activation640/quantization.
// No gate/up output is published. Separate scratch, same ordered FP32 sums.
// Mutable spans must be disjoint from each other and descriptor spans. Rejects
// null/count/alignment/overflow/alias before enqueue; caller validates device
// descriptor geometry, pointer extents and lifetimes.
// Optional stage_inputs is standalone-only input reuse; production callers
// retain the original false specialization.
// Optional gate_plan is a read-only standalone oracle: caller must supply every
// native batch exactly once, in original order, valid first/size and lifetime.
// Sparse planning/quantization still run; compact plan construction is excluded.
// transposed_scales requires a caller-owned, validated [40][1280][4] byte view
// per expert; original descriptors/scales remain necessary for decode/reference.
// kmajor_input_scales: producer directly writes
// [40][2*count][4] into the existing scale scratch; codes/math are unchanged.
// Standalone input staging also supports BOTH scale views with pitch32;
// single-view staging and pitch48 with views are rejected before enqueue.
// compact_logits is standalone/opt-in: lossless BF16-bit shared logit storage,
// requires both scale views, rejects staging/padding/skew/oracle plans.
// Opt-in token_input: caller MUST guarantee ALL gate/up globals equal
// token_global and all descriptor input pointers are complete contiguous rows
// of this caller-owned token_input[1..2048,2560]. Map is rebuilt each call;
// pointer/schema checks do not validate device descriptor contents. Requires
// compact logits and both SF views. token_indices has count writable ints.
cudaError_t launch_nvfp4_mma_activation_diagnostic(const Nvfp4Projection* jobs,
    const Nvfp4Projection* down_jobs,int count,Nvfp4MmaBatch* batches,
    uint8_t* packed_inputs,uint8_t* input_scales,float* activated,
    uint8_t* down_packed,uint8_t* down_scales,bool publish_activated,
    cudaStream_t stream=nullptr,bool stage_inputs=false,int input_pitch=32,bool pad_logits=false,bool skew_weights=false,
    const Nvfp4MmaBatch* gate_plan=nullptr,int gate_batches=0,bool transposed_scales=false,
    bool kmajor_input_scales=false,bool compact_logits=false,
    const float* token_input=nullptr,int token_count=0,float token_global=1.f,int* token_indices=nullptr,
    bool pair_words=false,bool weight_codes_kmajor=false,
    bool token_codes_kmajor=false,bool hoist_token_sources=false); // Hoisting diagnostic requires the current K-major token/weight path; no production selector.
// Diagnostic only, both choices include identical sentinel initialization.
cudaError_t launch_nvfp4_batch_plan_diagnostic(const Nvfp4Projection* jobs,int count,
    Nvfp4MmaBatch* batches,int columns,bool parallel,cudaStream_t stream=nullptr);
cudaError_t launch_nvfp4_mma_grouped(const Nvfp4Projection* jobs, int count,
    Nvfp4MmaBatch* batches, int rows, int inner, int split_row = 0,
    cudaStream_t stream = nullptr, uint8_t* packed_inputs = nullptr,
    uint8_t* input_scales = nullptr, int min_batch_size = 1,
    bool canonical_weights = false, bool prepare_singles = false,
    Nvfp4MmaBatch* compact_batches = nullptr, int compact_grid_y = 0,
    int max_batch_size = 8, bool specialized_prepared = false,
    bool inputs_prepared = false, bool async_weights = false,
    bool output_bf16 = false, bool early_inputs = false,
    bool parallel_plan = false,
    bool vector_quantize = false,
    bool gate_codes_kmajor = false);
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
