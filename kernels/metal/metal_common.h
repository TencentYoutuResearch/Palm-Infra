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

// RoPE over interleaved=false (Qwen3). Applies to [head_dim, S, heads].
struct RopeParams {
    int   head_dim;
    int   rope_dim;
    int   seq_len;
    int   heads;
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
    uint  a_offset;
    uint  b_offset;
    uint  out_offset;
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

// Fused Gated Delta Rule (GDN) linear-attention core for Qwen3.5.
// See kernels/gdn.h for the full algorithm/layout contract. All matmul-derived
// inputs are [seq, dim] row-major (ptr[t*dim+d]). State [num_v_heads, k_dim, v_dim]
// (state[vh*k_dim*v_dim + dk*v_dim + dv]) read+written in place.
//   qkv layout per token t: [ q(num_heads*k_dim) | k(num_heads*k_dim) | v(num_v_heads*v_dim) ]
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
