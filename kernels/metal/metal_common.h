#ifndef MOLLM_METAL_COMMON_H
#define MOLLM_METAL_COMMON_H

// Shared shader-side structs. Element strides/offsets are expressed in
// ELEMENTS (not bytes) — the host converts byte strides to element strides
// before filling these, because Metal buffer indexing is element-based once
// bound at an offset.

// Generic 4-D tensor descriptor for stride-aware elementwise / gather kernels.
// shape[0] = innermost (columns), matching the CPU Tensor layout.
struct TensorDesc {
    int   shape[4];
    int   stride[4];   // element strides
    uint  offset;      // element offset into the bound buffer
};

// CONCAT along dim 0: copy one source's dim-0 slab into a dense output.
// Source may be strided (view); output is dense over shape [out_shape0, s1, s2]
// (shape[3]==1, the MLA case). Dispatched once per concat input.
struct ConcatParams {
    int  shape[4];       // SOURCE shape
    int  stride[4];      // SOURCE element strides
    uint offset;         // SOURCE element offset
    int  dim_offset;     // running dim-0 offset of this source within the output
    int  out_shape0;     // total concatenated dim-0 (output dim-0 extent)
};

// Matmul params: C[M,N] = A[M,K] (fp32) * B[N,K]^T (fp16, row-major [N,K]).
// A is row-major [M,K]; B is stored row-major [N,K] (weights), so C[m,n] =
// sum_k A[m,k]*B[n,k]. Offsets in elements.
struct MatmulParams {
    int  M;
    int  N;
    int  K;
    uint a_offset;
    uint b_offset;
    uint c_offset;
    int  a_row_stride;   // elements between rows of A (>= K)
    int  b_row_stride;   // elements between rows of B (>= K)
    int  c_row_stride;   // elements between rows of C (>= N)
    int  activation;     // 0=NONE, 1=SILU (fused, phase-1 optional)
    int  act_n_begin;    // first output column receiving activation
    int  act_n_len;      // -1=all columns, 0=none, >0=range length
};

// W8 (int8 weight-only) matmul: C[M,N] = A[M,K](fp32) * W_int8[N,K] * scale_w.
// Weight is int8 row-major [N,K]; per-group fp32 scales at
// scale[n*groups_per_row + k/group_size]. Weight and scales are bound separately.
struct MatmulW8Params {
    int  M;
    int  N;
    int  K;
    uint a_offset;         // elements into A buffer
    uint c_offset;         // elements into C buffer
    int  a_row_stride;
    int  c_row_stride;
    int  activation;
    int  act_n_begin;
    int  act_n_len;
    int  groups_per_row;   // (K + group_size - 1) / group_size (w8pc => 1)
    int  group_size;       // K for per-channel (w8pc)
    // weight int8 and scales are bound at their own byte offsets (buffers 1, 4)
};

// W8A8 GEMM: int8 activations x int8 weights -> int32 -> dequant to fp32 C.
// Activations quantized per-token (row m) to int8 with scale_a[m]; weights are
// int8 [N,K] with per-channel scale_w[n]. out[m,n] = int32 * scale_a[m]*scale_w[n].
struct MatmulW8A8Params {
    int  M;
    int  N;
    int  K;
    uint c_offset;
    int  c_row_stride;
    int  activation;
    int  act_n_begin;
    int  act_n_len;
    // int8 A (buffer 0), int8 W (buffer 1), C (buffer 2), scale_a (buffer 4),
    // scale_w (buffer 5) bound at byte offsets; a is [M,K] contiguous (K inner).
};

// W4A8 GEMM: int8 activations x per-group symmetric int4 weights -> fp32 C.
// Weights packed [N, K/2] (byte = low nibble at even k, high nibble at odd k;
// w = nibble>=8 ? nibble-16 : nibble). Per-group fp32 scales scale_w[n*gpr+k/gs].
// Activations quantized per-token to int8 with scale_a[m]. out[m,n] =
// scale_a[m] * sum_g scale_w[n,g] * (int32 dot over group g).
struct MatmulW4A8Params {
    int  M;
    int  N;
    int  K;
    uint c_offset;
    int  c_row_stride;
    int  activation;
    int  act_n_begin;
    int  act_n_len;
    int  group_size;       // 128
    int  groups_per_row;   // K / group_size
    // int8 A (buffer 0), int4 W (buffer 1), C (buffer 2), scale_a (buffer 4),
    // scale_w (buffer 5) bound at byte offsets; A is [M,K] contiguous (K inner).
};

struct SelectedW4A8Params {
    int selections;
    int N;
    int K;
    int c_row_stride;
    int group_size;
    int groups_per_row;
    int rows_per_expert;
    int activation_rows;
    int activation_repeat;
};

// Per-token activation quantization: A_f32[M,K] -> A_i8[M,K] + scale_a[M].
// scale_a[m] = absmax(row m)/127; a_i8[m,k] = round(a[m,k]/scale_a[m]).
struct QuantActParams {
    int  M;
    int  K;
    uint a_offset;         // elements into fp32 A
    int  a_row_stride;     // elements between rows of A
    // out int8 A (buffer 2) contiguous [M,K]; scale_a (buffer 4) [M].
};

// RMS norm over dim0 (innermost). rows = product of dims 1..3.
struct RmsNormParams {
    int   dim0;          // normalized length
    int   rows;
    uint  x_offset;
    uint  w_offset;      // weight buffer offset
    uint  out_offset;
    int   x_row_stride;  // elements between rows of x
    int   out_row_stride;
    float eps;
};

// Layer norm over dim0 (innermost). rows = product of dims 1..3.
struct LayerNormParams {
    int   dim0;
    int   rows;
    uint  x_offset;
    uint  out_offset;
    int   x_row_stride;
    int   out_row_stride;
    float eps;
};

// RoPE over interleaved=false (Qwen3). Applies to [head_dim, S, heads].
struct RopeParams {
    int   head_dim;
    int   rope_dim;
    int   seq_len;
    int   heads;
    int   interleave;    // 1: pairs (0,1), 0: pairs (0, half)
    uint  x_offset;
    uint  cos_offset;
    uint  sin_offset;
    int   x_stride_pos;   // elements between positions
    int   x_stride_head;  // elements between heads
};

// Elementwise binary / unary.
struct EwiseParams {
    int   n;             // total elements (contiguous fast path)
    int   broadcast_b;   // 1 if b is scalar (nelements==1)
    int   shape0;        // logical inner width (for strided row views)
    int   a_row_stride;
    int   b_row_stride;
    int   out_row_stride;
    uint  a_offset;
    uint  b_offset;
    uint  out_offset;
    int   shape[4];
    int   a_stride[4];
    int   b_stride[4];   // zero on broadcast dimensions
    int   out_stride[4];
};

// Fused SwiGLU over a merged [2I, rows] tensor (gate = row[0..I), up = row[I..2I)).
// out[row*I + i] = silu(merged[row*mstride + i]) * merged[row*mstride + I + i].
struct SwigluParams {
    int   n;                 // total output elements (I * rows)
    int   I;                 // intermediate (half of merged dim0)
    uint  merged_offset;     // element offset into merged buffer
    uint  out_offset;        // element offset into output buffer
    int   merged_row_stride; // elements between rows (tokens) in merged (= 2I)
};

struct RwkvTokenShiftParams {
    int hidden;
    int seq;
    int real;
    int state_fp16;
    uint x_offset;
    uint state_offset;
    uint out_offset;
};

struct RwkvMixParams {
    int hidden;
    int total;
    uint x_offset;
    uint shift_offset;
    uint mix_offset;
    uint out_offset;
};

struct RwkvL2NormParams {
    int heads;
    int head_size;
    int groups;             // tokens * heads
    uint x_offset;
    uint out_offset;
    float eps;
};

struct RwkvPostParams {
    int heads;
    int head_size;
    int groups;
    uint raw_offset;
    uint r_offset;
    uint k_offset;
    uint v_offset;
    uint rk_offset;
    uint weight_offset;
    uint bias_offset;
    uint gate_offset;
    uint out_offset;
    float eps;
};

struct Rwkv7Params {
    int heads;
    int head_size;
    int seq;
    int real;
    int state_fp16;
    uint r_offset;
    uint decay_offset;
    uint k_offset;
    uint v_offset;
    uint a_offset;
    uint b_offset;
    uint state_offset;
    uint out_offset;
};

// Fused routed-expert MoE using decoded row-major W4-G128 expert weights.
// Route buffers are [seq,top_k]; merged is [seq,top_k,2*intermediate].
struct MoeW4Params {
    int hidden;
    int experts;
    int top_k;
    int intermediate;
    int seq_len;
    int n_group;
    int topk_group;
    int norm_topk;
    float routed_scale;
    uint hidden_offset;
    uint output_offset;
    int hidden_row_stride;
    int output_row_stride;
    int gu_groups_per_row;
    int down_groups_per_row;
};

// Fused Gated Delta Rule (GDN) linear-attention core for Qwen3.5.
// qkv split per token: [ q(num_heads*k_dim) | k(num_heads*k_dim) | v(num_v_heads*v_dim) ].
// Decode reads qkv as [seq,dim] (seq=1); prefill reads it as [qkv_total,seq]
// (dim-major), while a/b/z/out stay [seq,dim]. State [num_v_heads,k_dim,v_dim]
// (state[vh*k_dim*v_dim + dk*v_dim + dv]) is read and written in place.
struct GdnParams {
    int   num_heads;      // key heads (q/k/a/b)
    int   num_v_heads;    // value heads (v/z/out/state); repeat = num_v_heads/num_heads
    int   k_dim;
    int   v_dim;
    int   seq_len;
    int   n_real;         // real (non-padded) tokens; 0 = all
    int   use_qk_l2norm;
    float rms_eps;
    float l2_eps;
    float scale;
    // element offsets into each bound buffer
    uint  qkv_offset;
    uint  a_offset;
    uint  b_offset;
    uint  z_offset;
    uint  Alog_offset;
    uint  dtb_offset;
    uint  norm_offset;
    uint  state_offset;
    uint  out_offset;
};

// Depth-wise causal conv1d + silu (ShortConv). One thread per group.
//   x     : [groups, seq] with data layout [seq, groups] -> x[s*groups + g]
//   w     : [groups, kernel_size]                        -> w[g*ksize + k]
//   state : [groups, kernel_size-1] persistent, in-place -> state[g*(ksize-1) + p]
//   out   : [groups, seq]                                -> out[g*seq + i]
// window for position i = [state(ksize-1) | x[0..seq)]; out[i]=silu(sum win[i+k]*w[k]).
// state after: last (ksize-1) real x values.
struct ShortConvParams {
    int  groups;
    int  seq;
    int  kernel_size;      // e.g. 4
    int  n_real;           // real (non-padded) positions to process
    uint x_offset;         // element offsets into each bound buffer
    uint w_offset;
    uint state_offset;
    uint out_offset;
};

// KV-cache append: copy K_cur/V_cur (FP32) rows into the FP16 cache at
// position (past + s), per kv-head. cache element offset already accounts for
// the 64-byte CacheMetadata header (added on the host as a byte offset).
struct SdpaAppendParams {
    int  num_kv_heads;
    int  cur_seqlen;
    int  past_seqlen;
    int  head_dim;       // width of the row being copied (head_dim or v_head_dim)
    int  max_seq_len;    // cache capacity (stride between kv-heads = head_dim*max_seq_len)
    uint cur_offset;     // element offset into K_cur/V_cur buffer
    int  cur_stride_head;// elements between kv-heads in K_cur/V_cur (stride[2])
    int  cur_stride_pos; // elements between positions in K_cur/V_cur (stride[1])
    uint cache_offset;   // element offset into cache buffer (past the metadata header)
};

// SDPA compute: one threadgroup per (head, query position). Loops over all
// dst_seqlen keys; causal masking uses (past + query_pos). FP16 cache.
struct SdpaParams {
    int   num_heads;
    int   num_kv_heads;
    int   head_dim;
    int   v_head_dim;
    int   src_seqlen;
    int   dst_seqlen;      // past + cur
    int   past_seqlen;
    int   max_seq_len;     // cache capacity (per-head stride)
    int   causal;
    float scale;
    uint  q_offset;        // element offset into Q buffer
    int   q_stride_pos;    // Q stride[1] in elements
    int   q_stride_head;   // Q stride[2] in elements
    uint  k_cache_offset;  // element offset into K cache buffer (past header)
    uint  v_cache_offset;  // element offset into V cache buffer (past header)
    uint  o_offset;        // element offset into output buffer
    int   o_stride_pos;    // out stride[1]
    int   o_stride_head;   // out stride[2]
    uint  mask_offset;     // element offset into mask buffer (0 if no mask)
    int   has_mask;
    int   mask_stride_row; // mask stride[1] in elements (row = query pos)
};

#endif
