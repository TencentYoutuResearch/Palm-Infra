#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mollm_cuda {
struct Nvfp4Projection;
// Diagnostic: one token, canonical descriptors, 1..16 routes. Output is
// ordered sum of BF16-rounded projections, without rounding the sum itself.
cudaError_t launch_nvfp4_route_sum(const Nvfp4Projection* jobs, int count,
    int rows, int inner, float* summed_output, cudaStream_t stream = nullptr,
    int rows_per_cta = 1);
// Diagnostic only: caller supplies canonical row/pair weights (never MMA)
// and block_scales in [row/4][K/16][row%4] order; rows%4==0.
cudaError_t launch_nvfp4_tiled_scale_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);
struct Nvfp4MmaBatch;

// One routed expert projection. Pointers may refer to coherent system memory
// only after the backend has checked pageable-memory/host-page-table support.
struct Nvfp4Projection {
    const uint8_t* weight;
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

// Diagnostic canonical layouts only: preserve original 32-lane FMA chains/tree.
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

void launch_nvfp4_projections(const Nvfp4Projection* jobs, int count,
                              int rows, int inner, cudaStream_t stream = nullptr,
                              bool mma_layout = false);
// Diagnostic: pair adjacent equal-expert jobs inside projection, no planning pass.
cudaError_t launch_nvfp4_inline_pairs(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream = nullptr);
// Diagnostic: use pair ranks emitted by grouping; count entries, no extra planner.
cudaError_t launch_nvfp4_ranked_pairs(const Nvfp4Projection* jobs, int count,
    const Nvfp4MmaBatch* pair_ranks, int rows, int inner, cudaStream_t stream = nullptr);
// Original SIMT iteration lookahead; canonical row/pair layouts only.
// Diagnostic gate/up+activation fusion; canonical jobs, width multiple of four.
// Diagnostic register input reuse across 2/4 output rows, original per-row tree.
cudaError_t launch_nvfp4_row_reuse(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int row_group, cudaStream_t stream = nullptr);
cudaError_t launch_nvfp4_gate_activation(const Nvfp4Projection* jobs, int count,
    int width, int inner, const float* route_weights, float* activated,
    cudaStream_t stream = nullptr, int warps = 4, int load_hint = 0);
// Preserves FP32 accumulation/reduction order. No allocation or synchronization.
cudaError_t launch_nvfp4_pipelined_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, bool pipeline_inputs = true, cudaStream_t stream = nullptr,
    bool cache_input = false, int load_hint = 0, bool direct_fp4 = false,
    bool fixed_inner = false, bool warp_packet = false,
    bool aligned_packets = false, bool direct_scale = false, bool route_major = false,
    bool pair_only = false, bool broadcast_scale = false); // Broadcast requires K%128==0. Pair-only caller guarantees canonical pair layout; aligned caller guarantees 2-byte weights / 16-byte inputs.
// Diagnostic native fragment-plane layout, one-step SIMT weight/input pipeline.
// Weights/scales must have padded ceil(rows/16)*16 allocation as for repacking.
cudaError_t launch_nvfp4_native_pipelined_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream = nullptr);
// Diagnostic: caller supplies two device arrays of 256 FP32 bit patterns.
cudaError_t launch_nvfp4_scale_decode_check(uint32_t* hardware, uint32_t* direct,
    cudaStream_t stream = nullptr);
// Diagnostic only: weight points to 36-byte four-row/group tiles, containing
// 32 FP4 bytes then four E4M3 scales. block_scales is unused. No ABI change.
cudaError_t launch_nvfp4_interleaved_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream = nullptr);
// Diagnostic only: vary CTA size without changing the per-row arithmetic.
cudaError_t launch_nvfp4_cta_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int warps, cudaStream_t stream = nullptr);
// Diagnostic only: full-input lookahead with depth 2 or 4. Same canonical ABI.
cudaError_t launch_nvfp4_deep_pipelined_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int depth, cudaStream_t stream = nullptr);
// Diagnostic only: adjacent equal-weight jobs share dequantization. Unequal
// pairs and an odd last job remain valid. Original FP32 accumulation order.
cudaError_t launch_nvfp4_paired_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream = nullptr, bool mma_layout = false);
// Reuse the routed descriptor ABI for FP16 expert-major matrices. Only
// weight/input/output fields are read; route weights are applied by activation.
cudaError_t launch_fp16_expert_projections(const Nvfp4Projection* jobs,
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
    bool gate_codes_kmajor = false);
// gate_codes_kmajor requires canonical_weights and fixed1280x2560 code view;
// block/row scales remain canonical. Default false preserves old dispatch.
// Optional caller-owned scratch prepares each job once; parts must be one.
// Capacity: count*(split_row?2:1)*inner/2 packed bytes and inner/16 scale bytes.
// Diagnostic only: prefetch weights/scales without changing arithmetic.
// Diagnostic batched N8: consecutive groups of eight descriptors MUST share
// native weight, block-scale and row-scale pointers. Inputs/globals/output
// weights may differ. No padding, sorting, allocation or split-K is performed.
cudaError_t launch_nvfp4_mma_batched8(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int split_row = 0, cudaStream_t stream = nullptr);
struct Nvfp4MmaBatch { int first; int size; };
// Diagnostic SIMT pairing within adjacent equal-weight runs. Scratch holds
// count+1 entries. Unpaired jobs retain the independent projection arithmetic.
// Planning and both projection launches are stream ordered and graph safe.
cudaError_t launch_nvfp4_grouped_pairs(const Nvfp4Projection* jobs, int count,
    Nvfp4MmaBatch* plan, int rows, int inner, cudaStream_t stream = nullptr,
    bool mma_layout = false);
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
    bool gate_codes_kmajor = false); // Already prepared input requires split_row==0.
// gate_codes_kmajor: all jobs use fixed1280x2560 K64-major gate code views,
// canonical scales/weights required; default false retains old dispatch.
// Optional vector_quantize: supported BF16 async gate only; scalar max/order.
// parallel_plan is sparse only; compact/specialized plans are unsupported.
// Diagnostic early_inputs is restricted to the supported BF16 async gate.
// Optional output_bf16: canonical packed sparse down2560/K640/N8/minimum1,
// split0, four warps, no async/specialized/single fallback. Descriptor output
// pointers then reference rows uint16_t codes, not rows floats. Codes use the
// existing bit-level BF16 boundary, including NaN truncation. Ordinary kernels
// must never consume these output descriptors/buffers.
// Also supports canonical async N32 gate1280/K2560/split640/minimum1 with
// four warps and no compact/specialized/single fallback. Pair only with the
// BF16-input activation preparation consumer, never ordinary FLOAT readers.
// Diagnostic async_weights: canonical sparse N32 gate1280/K2560/four warps,
// weight bases aligned16 and block scales aligned4; no layout or backend change.
// max_batch_size is 8 (original kernel), 16, or experimental 32.
// The wider kernels require prequantization scratch and reuse each weight
// fragment for two/four N8 MMAs at width16/32 respectively.
// specialized_prepared also selects the prepared-only specialization at width8;
// it requires packed scratch. The original width8 default is unchanged.
// Optional compact_batches holds count+1 entries, separate from all other
// scratch. Its first entry stores a device-side batch count. Compact CTAs
// process this plan without a host readback. compact_grid_y is a performance
// hint in [0,count]; zero selects ceil(count/max_batch_size). An overflow kernel
// handles arbitrary run counts, so an underestimated hint cannot drop work.
// Original route/output indices are preserved.
// canonical_weights selects existing row/pair storage without repacking.
// prepare_singles requires packed scratch; quantizes small groups once and
// reuses those inputs in their scalar fallback instead of per-CTA quantization.
// Groups require identical weight, scale pointers and layout metadata.
// min_batch_size in [1,8]: smaller groups use separate matching projections;
// optional prequantization skips these groups. The default scalar kernel
// remains a separate, unfiltered specialization.
// Optional prequantization scratch capacities: count*(split_row?2:1)*inner/2
// packed bytes and count*(split_row?2:1)*inner/16 scale bytes, CUDA aligned.
void launch_nvfp4_prefetch_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int iterations, cudaStream_t stream = nullptr);
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
// Independent diagnostic: scalar/subgroup quantization, disjoint buffers.
// Descriptor-order output; optional batches filter each variant identically.
cudaError_t launch_nvfp4_quantize_probe(const Nvfp4Projection* jobs,int count,int inner,
    int variants,uint8_t* packed,uint8_t* scales,const Nvfp4MmaBatch* batches,
    int minimum,int mode,cudaStream_t stream = nullptr); //0 scalar;1 sixteen lanes;2 eight lanes;3 shared scalar max;4 vector I/O.
// Diagnostic only: maxima scratch holds input_rows*(inner/16) floats.
// Recompute raw maxima for contiguous FP32 input rows, then quantize each
// descriptor/global independently. Unmatched/shifted input pointers fall back
// to scalar maxima. Includes the statistics launch; no inference selector.
cudaError_t launch_nvfp4_quantize_cached_max_probe(const Nvfp4Projection* jobs,
    int count, int inner, int variants, uint8_t* packed, uint8_t* scales,
    const Nvfp4MmaBatch* batches, int minimum, const float* input,
    int input_rows, float* maxima, cudaStream_t stream = nullptr);

}  // namespace mollm_cuda
