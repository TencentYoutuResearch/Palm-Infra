"""
mollm model builder for Qwen3.5-0.8B (text only).

Hybrid architecture:
  - 18 layers of linear attention (Gated Delta Rule)
  - 6 layers of full attention (GQA + QK norm + output gate)
  - Standard SwiGLU MLP

Text-only: vision encoder weights are ignored.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

from transpile import GraphBuilder, Precision, _write_weight_file, quantize_weight_w8_group, save_package


def _quant_group_size(quant: str, k: int) -> int | None:
    if quant == "none":
        return None
    if quant == "w8pc":
        return k
    if quant.startswith("w8g"):
        return int(quant[3:])
    raise ValueError(f"unsupported quant mode: {quant}")


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    """Load a single-file safetensors model."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))

    tensors = {}
    with open(path, 'rb') as f:
        f.seek(8 + header_len)
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dtype_str = meta['dtype']
            shape = meta['shape']
            offsets = meta['data_offsets']
            f.seek(8 + header_len + offsets[0])
            data = f.read(offsets[1] - offsets[0])
            np_dtype = {
                'F32': np.float32, 'F16': np.float16,
                'BF16': np.uint16,
            }[dtype_str]
            arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            if dtype_str == 'F16':
                arr = arr.astype(np.float16)
            elif dtype_str == 'BF16':
                as_u32 = arr.astype(np.uint32) << 16
                arr = as_u32.view(np.float32)
            tensors[name] = arr
    return tensors


def export_weights(weights: dict, weights_dir: str, quant: str = "none"):
    """Export text model weights. Skip vision encoder."""
    os.makedirs(weights_dir, exist_ok=True)

    def save(name: str, data: np.ndarray, quantizable: bool = False):
        wpath = os.path.join(weights_dir, f"{name}.weights")
        group_size = _quant_group_size(quant, data.shape[1]) if quantizable and data.ndim == 2 else None
        if group_size is not None:
            q, scales, gs, ng = quantize_weight_w8_group(data, group_size)
            _write_weight_file(wpath, q, scales=scales, group_size=gs, num_groups=ng)
        else:
            _write_weight_file(wpath, data)

    # Skip all visual/vision weights
    for wname, wdata in weights.items():
        if 'visual' in wname or 'vision' in wname:
            continue
        if 'language_model' not in wname:
            continue

        # Final norm (check before generic norm)
        if wname.endswith('model.language_model.norm.weight'):
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            d = 1.0 + d  # Qwen3_5RMSNorm: output * rms(x) * (1.0 + weight)
            save('final_norm', d)
            continue

        # RMSNorm weights → FP32.
        # Qwen3_5RMSNorm (input_layernorm, post_attention_layernorm, q_norm, k_norm) uses:
        #   output * (1.0 + weight)  →  pre-compute (1.0 + weight)
        # Qwen3_5RMSNormGated (linear_attn.norm) uses:
        #   output * weight  →  standard RMSNorm, no conversion needed
        if 'norm' in wname.lower() or 'layernorm' in wname.lower():
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            # Add 1.0 for all Qwen3_5RMSNorm variants
            # (input_layernorm, post_attention_layernorm, q_norm, k_norm)
            # but NOT for linear_attn.norm which is RMSNormGated
            if 'linear_attn' not in wname:
                d = 1.0 + d
            save(wname.replace('.', '_'), d)
            continue

        # A_log, dt_bias → keep as-is (already F32 for A_log, BF16→F32 for dt_bias)
        if 'A_log' in wname or 'dt_bias' in wname:
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            save(wname.replace('.', '_'), d)
            continue

        # Embed tokens → FP16
        if 'embed_tokens' in wname:
            d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
            save('embed_tokens', d)
            continue

        # conv1d.weight → FP32 (used by ShortConv scalar kernel, needs FP32)
        if 'conv1d' in wname:
            d = wdata.reshape(wdata.shape[0], wdata.shape[2])  # [6144, 1, 4] → [6144, 4]
            d = d.astype(np.float32) if d.dtype != np.float32 else d
            save(wname.replace('.', '_'), d)
            continue

        # Projection weights are FP16 by default, optionally W8 quantized.
        d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
        save(wname.replace('.', '_'), d, quantizable=True)


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096, is_prefill: bool = False) -> GraphBuilder:
    """Build prefill or decode graph for Qwen3.5 text model.

    When is_prefill=True, the hidden INPUT's seq dim is marked DynamicKind.SEQ
    so the C++ runtime can inject actual seq_len (dynamic shape mode).
    Decode graphs (seq=1) stay all-STATIC.
    """
    g = GraphBuilder()
    tc = cfg['text_config'] if 'text_config' in cfg else cfg

    hidden_size = tc['hidden_size']
    num_layers = tc['num_hidden_layers']
    layer_types = tc['layer_types']
    eps = tc.get('rms_norm_eps', 1e-6)
    rope_theta = tc['rope_parameters']['rope_theta']
    rope_dim = int(tc['head_dim'] * tc['rope_parameters']['partial_rotary_factor'])  # 256 * 0.25 = 64

    # Full attention params
    num_heads = tc['num_attention_heads']           # 8
    num_kv_heads = tc['num_key_value_heads']        # 2
    head_dim = tc['head_dim']                       # 256

    # Linear attention params
    linear_num_heads = tc['linear_num_key_heads']   # 16
    linear_k_dim = tc['linear_key_head_dim']         # 128
    linear_v_dim = tc['linear_value_head_dim']      # 128
    linear_num_v_heads = tc.get('linear_num_value_heads', linear_num_heads)  # 0.8B=16, 4B=32
    conv_kernel = tc['linear_conv_kernel_dim']       # 4

    # MLP
    intermediate = tc.get('intermediate_size', 3584)

    print(f"Qwen3.5 graph: seq_len={seq_len}, layers={num_layers}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"linear_heads={linear_num_heads}, linear_v_heads={linear_num_v_heads}, "
          f"linear_k={linear_k_dim}, linear_v={linear_v_dim}, conv_kernel={conv_kernel}")

    # ---- set graph metadata (engine reads these instead of CLI args) ----
    g.set_model_config(
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=tc['vocab_size'],
        model_type='qwen3_5',
    )

    # ---- embed_tokens ----
    embed_path = os.path.join(weights_dir, "embed_tokens.weights")
    embed_shape = (tc['vocab_size'], hidden_size)
    g.weight(embed_path, embed_shape, Precision.FP16)

    # ---- graph inputs ----
    # In prefill graphs, mark seq dim (shape[1]) as SEQ so the C++ runtime
    # can substitute actual seq_len. Decode graphs stay all-CONST.
    if is_prefill:
        from transpile import DimExpr
        SEQ_DIM = DimExpr.seq()
        CONST   = DimExpr.const()
        hidden_dyn = (CONST, SEQ_DIM, CONST, CONST)
        mask_dyn   = (CONST, SEQ_DIM, CONST, CONST)
        cos_dyn    = (CONST, SEQ_DIM, CONST, CONST)
        sin_dyn    = (CONST, SEQ_DIM, CONST, CONST)
    else:
        hidden_dyn = mask_dyn = cos_dyn = sin_dyn = None
    hidden = g.input('hidden', (hidden_size, seq_len), dynamic=hidden_dyn)
    mask = g.input('mask', (1, seq_len), dynamic=mask_dyn)
    cos = g.input('cos', (rope_dim // 2, seq_len), dynamic=cos_dyn)
    sin = g.input('sin', (rope_dim // 2, seq_len), dynamic=sin_dyn)

    # ---- persistent state inputs (KV cache for full attn, GDN state for linear attn) ----
    cache_inputs = []
    for i in range(num_layers):
        lt = layer_types[i]
        if lt == 'full_attention':
            ck = g.input(f'cache_k{i}', (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cv = g.input(f'cache_v{i}', (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cache_inputs.append(('kv', ck, cv))
        else:
            # GDN recurrent state: [v_dim, k_dim, num_value_heads] FP32
            gs = g.input(f'gdn_state{i}', (linear_v_dim, linear_k_dim, linear_num_v_heads), prec=Precision.FP32)
            # Conv state: [qkv_total, conv_kernel-1] FP16
            qkv_total = linear_num_heads * linear_k_dim * 2 + linear_num_v_heads * linear_v_dim
            gc = g.input(f'gdn_conv{i}', (qkv_total, conv_kernel - 1), prec=Precision.FP16)
            cache_inputs.append(('gdn', gs, gc))

    # ---- build layers ----
    x = hidden
    for i in range(num_layers):
        lt = layer_types[i]
        ck_in, cv_in = (cache_inputs[i][1], cache_inputs[i][2]) if cache_inputs[i][0] == 'kv' else (None, None)
        gs_in, gc_in = (cache_inputs[i][1], cache_inputs[i][2]) if cache_inputs[i][0] == 'gdn' else (None, None)

        x = _build_layer(g, x, i, weights_dir, cfg, tc,
                        cos, sin, mask, ck_in, cv_in, gs_in, gc_in,
                        eps, seq_len, rope_dim, rope_theta,
                        num_heads, num_kv_heads, head_dim,
                        linear_num_heads, linear_k_dim, linear_v_dim,
                        linear_num_v_heads,
                        conv_kernel, intermediate, hidden_size, lt,
                        is_prefill=is_prefill)

    # ---- final norm ----
    w_norm = g.weight(os.path.join(weights_dir, "final_norm.weights"),
                      (hidden_size,), Precision.FP32)
    x = g.rms_norm(x, w_norm, eps=eps)

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def _build_layer(g, x, layer_idx, weights_dir, cfg, tc,
                 cos, sin, mask, ck_in, cv_in, gs_in, gc_in,
                 eps, seq_len, rope_dim, rope_theta,
                 num_heads, num_kv_heads, head_dim,
                 linear_num_heads, linear_k_dim, linear_v_dim,
                 linear_num_v_heads,
                 conv_kernel, intermediate, hidden_size, layer_type,
                 is_prefill=False):
    """Build one layer (linear_attention or full_attention)."""

    pfx_lm = f'model_language_model_layers_{layer_idx}'

    # ---- Input RMSNorm ----
    w_ln = g.weight(os.path.join(weights_dir, f"{pfx_lm}_input_layernorm_weight.weights"),
                    (hidden_size,), Precision.FP32)
    x_normed = g.rms_norm(x, w_ln, eps=eps)

    if layer_type == 'linear_attention':
        attn_out = _build_linear_attn_layer(g, x_normed, layer_idx, weights_dir,
                                             gs_in, gc_in, eps, seq_len,
                                             linear_num_heads, linear_k_dim, linear_v_dim,
                                             linear_num_v_heads,
                                             conv_kernel, hidden_size)
    else:
        attn_out = _build_full_attn_layer(g, x_normed, layer_idx, weights_dir,
                                           cos, sin, mask, ck_in, cv_in,
                                           eps, seq_len, rope_dim,
                                           num_heads, num_kv_heads, head_dim,
                                           hidden_size, is_prefill=is_prefill)

    # ---- Residual ----
    x = g.add(x, attn_out)

    # ---- Post-attention RMSNorm ----
    w_ln2 = g.weight(os.path.join(weights_dir, f"{pfx_lm}_post_attention_layernorm_weight.weights"),
                     (hidden_size,), Precision.FP32)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    # ---- MLP (SwiGLU) ----
    mlp_pfx = f'{pfx_lm}_mlp'
    w_gate = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_proj_weight.weights"),
                      (intermediate, hidden_size), Precision.FP16)
    w_up = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_up_proj_weight.weights"),
                    (intermediate, hidden_size), Precision.FP16)
    w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_down_proj_weight.weights"),
                      (hidden_size, intermediate), Precision.FP16)

    gate = g.matmul(x_normed2, w_gate)
    up = g.matmul(x_normed2, w_up)
    gate = g.silu(gate)
    mlp_hidden = g.mul(gate, up)
    mlp_out = g.matmul(mlp_hidden, w_down)
    x = g.add(x, mlp_out)

    return x


def _build_linear_attn_layer(g, x, layer_idx, weights_dir,
                              gs_in, gc_in, eps, seq_len,
                              num_heads, k_dim, v_dim, num_v_heads,
                              conv_kernel, hidden_size):
    """Build a linear attention (Gated Delta Rule) layer.

    Uses the fused `gated_deltanet` op for the GDN core + RMSNormGated.
    See kernels/gdn.h for the op's data-layout contract — all matmul-derived
    inputs are consumed in their native [seq, dim] row-major data layout.
    """
    pfx = f'model_language_model_layers_{layer_idx}_linear_attn'

    # ---- Projections ----
    w_qkv = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_qkv_weight.weights"),
                     (num_heads * k_dim * 2 + num_v_heads * v_dim, hidden_size), Precision.FP16)
    qkv = g.matmul(x, w_qkv)  # shape [qkv_total, seq], data [seq, qkv_total]

    w_a = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_a_weight.weights"),
                   (num_v_heads, hidden_size), Precision.FP16)
    a_out = g.matmul(x, w_a)  # shape [num_v_heads, seq], data [seq, num_v_heads]

    w_b = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_b_weight.weights"),
                   (num_v_heads, hidden_size), Precision.FP16)
    b_out = g.matmul(x, w_b)  # shape [num_v_heads, seq], data [seq, num_v_heads]

    w_z = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_z_weight.weights"),
                   (num_v_heads * v_dim, hidden_size), Precision.FP16)
    z_out = g.matmul(x, w_z)  # shape [num_v_heads*v_dim, seq], data [seq, num_v_heads*v_dim]

    # ---- A_log and dt_bias (per-value-head constants for 4B, per-key-head for 0.8B) ----
    A_log = g.weight(os.path.join(weights_dir, f"{pfx}_A_log.weights"),
                     (num_v_heads,), Precision.FP32)
    dt_bias = g.weight(os.path.join(weights_dir, f"{pfx}_dt_bias.weights"),
                       (num_v_heads,), Precision.FP32)

    # ---- Short conv on qkv (conv1d + silu) ----
    # matmul output shape [qkv_total, seq], data [seq, qkv_total] (matmul convention).
    # ShortConv kernel reads data as [seq, groups] (x_data[s*groups + g]), so it
    # consumes the matmul output directly — no reshape/permute needed.
    # conv1d.weight: [6144, 1, 4] → reshape to [6144, 4] (groups=6144, kernel_size=4)
    qkv_total = num_heads * k_dim * 2 + num_v_heads * v_dim
    w_conv = g.weight(os.path.join(weights_dir, f"{pfx}_conv1d_weight.weights"),
                     (qkv_total, conv_kernel), Precision.FP32)
    qkv_conv = g.shortconv(qkv, w_conv, gc_in, kernel_size=conv_kernel)
    # qkv_conv data is [qkv_total, seq] (shortconv materializes [groups, seq]).
    # The fused kernel reads this layout directly — do NOT reshape (mollm reshape
    # is metadata-only and would not transpose the data).

    # ---- Fused GDN core + RMSNormGated ----
    # Replaces: split qkv, g/beta compute, GDN recurrence, RMSNormGated chain.
    # Output: shape [num_v_heads*v_dim, seq], data [seq, num_v_heads*v_dim] row-major.
    w_norm = g.weight(os.path.join(weights_dir, f"{pfx}_norm_weight.weights"),
                     (v_dim,), Precision.FP32)
    gated = g.gated_deltanet(
        qkv_conv, a_out, b_out, z_out,
        A_log, dt_bias, w_norm, gs_in,
        num_heads=num_heads, k_dim=k_dim, v_dim=v_dim, seq_len=seq_len,
        use_qk_l2norm=True, rms_eps=eps,
        num_v_heads=num_v_heads)

    # ---- out_proj ----
    w_out = g.weight(os.path.join(weights_dir, f"{pfx}_out_proj_weight.weights"),
                     (hidden_size, num_v_heads * v_dim), Precision.FP16)
    out = g.matmul(gated, w_out)
    return out


def _build_full_attn_layer(g, x, layer_idx, weights_dir,
                            cos, sin, mask, ck_in, cv_in,
                            eps, seq_len, rope_dim,
                            num_heads, num_kv_heads, head_dim, hidden_size,
                            is_prefill=False):
    """Build a full attention (GQA + QK norm + output gate) layer."""
    # In prefill graphs, use SEQ symbol for seq dims (runtime substitutes
    # actual seq_len). In decode graphs, use seq_len literal (=1, static).
    from transpile import SEQ
    _S = SEQ.bind(seq_len) if is_prefill else seq_len
    pfx = f'model_language_model_layers_{layer_idx}_self_attn'

    # ---- q_proj: [4096, 1024] → 8 × (256 query + 256 gate) ----
    # q_proj output is chunked: first half = query, second half = gate
    w_q = g.weight(os.path.join(weights_dir, f"{pfx}_q_proj_weight.weights"),
                   (num_heads * head_dim * 2, hidden_size), Precision.FP16)
    qg = g.matmul(x, w_q)  # [8 × 512, seq] = [4096, 4], data [4, 4096]

    # ---- Split query and gate (matching HF's view+chunk) ----
    # HF: q_proj_out.view(seq, NH, HD*2).chunk(2, dim=-1)
    #   In row-major [seq, NH, HD*2]: (s, h, d2) at s*4096 + h*512 + d2.
    #   query = d2[0:256], gate = d2[256:511].
    #
    # To match this in C++:
    #   qg [4096,4], data [4,4096] → contiguous → reshape [512, 8, 4]
    #   In [512, 8, 4] row-major: (d2, h, s) at d2 + h*512 + s*4096.
    #   Same as HF! d2 + h*512 + s*4096 = s*4096 + h*512 + d2. ✓
    #   slice dim=0: query = [256, 8, 4] (d2=0..255), gate = [256, 8, 4] (d2=256..511)
    #   reshape query → [2048, 4]: (dim, s) at dim + s*2048.
    #     dim = d + h*256, so (s, h, d) at s*2048 + h*256 + d. ✓
    #   reshape gate → [2048, 4]: same layout. ✓
    # No contiguous() needed: reshape + slice both use stride-aware accessors,
    # and the matmul output's [seq, NH*HD*2] row-major layout already matches
    # the [HD*2, NH, seq] view (same element ordering, see comment above).
    qg = g.reshape(qg, (head_dim * 2, num_heads, _S))        # [512, 8, seq]
    query, gate = g.slice(qg, [head_dim, head_dim], dim=0)         # [256, 8, seq] each
    query = g.reshape(query, (num_heads * head_dim, _S))      # [2048, seq]
    gate = g.reshape(gate, (num_heads * head_dim, _S))        # [2048, seq]

    # ---- k_proj, v_proj ----
    w_k = g.weight(os.path.join(weights_dir, f"{pfx}_k_proj_weight.weights"),
                   (num_kv_heads * head_dim, hidden_size), Precision.FP16)
    w_v = g.weight(os.path.join(weights_dir, f"{pfx}_v_proj_weight.weights"),
                   (num_kv_heads * head_dim, hidden_size), Precision.FP16)
    k = g.matmul(x, w_k)
    v = g.matmul(x, w_v)

    # ---- QK norm (RMSNorm per head) ----
    # q_norm/k_norm weight shape: [head_dim]
    # query/k are matmul outputs: declared [NH*HD, seq], data [seq, NH*HD] row-major.
    #
    # The RMSNorm kernel normalizes over dim0 (contiguous elements). We need
    # each (head, token) pair's head_dim=256 elements to be contiguous and
    # grouped into one "column" for RMSNorm.
    #
    # Correct pipeline:
    #   1. contiguous(matm_out) → reshape(HD, NH, seq)
    #      Matmul data layout [seq, NH*HD] matches [HD, NH, seq] row-major:
    #      element (d, h, s) at d + h*HD + s*HD*NH.
    #   2. permute(HD, NH, seq) → (HD, seq, NH) → contiguous
    #      Materializes as [HD, seq, NH] row-major. Each (s, h) pair's 256
    #      elements are at the correct positions.
    #   3. reshape(HD, seq, NH) → (HD, NH*seq)
    #      Columns: k = s + h*seq_len. This is s-major interleaved:
    #      s0_h0, s1_h0, s2_h0, s3_h0, s0_h1, s1_h1, ...
    #      Each column has the correct (s, h) pair's 256 elements.
    #   4. RMSNorm over dim0
    #   5. reshape(HD, NH*seq) → (HD, seq, NH)
    #      Maps column k back to (d, s=k%seq_len, h=k/seq_len).
    #      Since columns are s-major, this gives correct (s, h) pairs.
    #   6. contiguous → RoPE/SDPA expects [HD, seq, NH] row-major.
    w_qn = g.weight(os.path.join(weights_dir, f"{pfx}_q_norm_weight.weights"),
                    (head_dim,), Precision.FP32)
    w_kn = g.weight(os.path.join(weights_dir, f"{pfx}_k_norm_weight.weights"),
                    (head_dim,), Precision.FP32)

    # --- Query ---
    # No contiguous() before reshape/permute: reshape inherits stride,
    # permute is zero-copy. The downstream rms_norm kernel reads via
    # stride[1] (ldx), so it handles the strided permuted view directly.
    query = g.reshape(query, (head_dim, num_heads, _S))  # [HD, NH, seq]
    query = g.permute(query, (0, 2, 1, 3))                   # [HD, seq, NH]
    # No contiguous() before rms_norm: kernel_rms_norm uses ldx=stride[1].
    query = g.reshape(query, (head_dim, num_heads * _S)) # [HD, NH*seq], cols s-major
    query = g.rms_norm(query, w_qn, eps=eps)
    query = g.reshape(query, (head_dim, _S, num_heads))  # [HD, seq, NH]
    query = g.contiguous(query)                                # for RoPE/SDPA

    # --- Key ---
    # No contiguous() before reshape/permute/rms_norm: same rationale as Query.
    k = g.reshape(k, (head_dim, num_kv_heads, _S))       # [HD, NKV, seq]
    k = g.permute(k, (0, 2, 1, 3))                            # [HD, seq, NKV]
    k = g.reshape(k, (head_dim, num_kv_heads * _S))      # [HD, NKV*seq], cols s-major
    k = g.rms_norm(k, w_kn, eps=eps)
    k = g.reshape(k, (head_dim, _S, num_kv_heads))       # [HD, seq, NKV]
    k = g.contiguous(k)                                         # for RoPE/SDPA
    # v is matmul output: declared shape [nkv*hd, seq], data [seq, nkv*hd] row-major.
    # Step 1: reshape (hd, nkv, seq) — zero-copy, d0=hd innermost.
    #   flat[0..hd-1] = (d=0..hd-1, nkv=0, s=0) = (s=0, nkv=0, d=0..hd-1) ✓
    # Step 2: permute (hd, nkv, seq) → (hd, seq, nkv)
    # Step 3: contiguous → SDPA-expected [hd, seq, nkv] row-major
    # No contiguous() before reshape/permute: reshape+permute are zero-copy
    # stride-aware; the single contiguous() before SDPA materializes the
    # layout SDPA requires (stride[1] == hd*es).
    v = g.reshape(v, (head_dim, num_kv_heads, _S))
    v = g.permute(v, (0, 2, 1, 3))
    v = g.contiguous(v)

    # ---- RoPE (partial: only rope_dim=64 out of head_dim=256) ----
    # kernel_rope applies rotation to [0, rope_dim) and copies [rope_dim, D) unchanged.
    # Qwen3.5 uses rotate_half (halves mode), NOT interleave.
    query = g.rope(query, cos, sin, rope_dim=rope_dim, interleave=False)
    k = g.rope(k, cos, sin, rope_dim=rope_dim, interleave=False)

    # ---- SDPA ----
    scale = head_dim ** -0.5
    attn, ck_out, cv_out = g.sdpa(
        query, k, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=scale,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, v_head_dim=head_dim)

    # ---- Reshape attn from [head_dim, seq, num_heads] → [num_heads*head_dim, seq] ----
    # SDPA output is [hd, seq, nheads] row-major: element (d, s, h) at d + s*hd + h*hd*seq.
    # To get [NH*HD, seq] with correct (s,h,d) ordering:
    #   permute [HD, seq, NH] → [HD, NH, seq] (zero-copy)
    #   contiguous → materialize [HD, NH, seq] row-major
    #   reshape [HD, NH, seq] → [NH*HD, seq] (zero-copy)
    # In [HD, NH, seq] row-major: (d, h, s) at d + h*HD + s*HD*NH.
    # Reshape: (dim, s) at dim + s*NH*HD, where dim = d + h*HD.
    #   flat[s*2048 + h*256 + d] = SDPA(s, h, d). ✓
    attn = g.permute(attn, (0, 2, 1, 3))     # [HD, NH, seq]
    attn = g.contiguous(attn)                  # materialize
    attn = g.reshape(attn, (num_heads * head_dim, _S))  # [NH*HD, seq]

    # ---- Output gate: attn * sigmoid(gate) ----
    # gate is matmul output (qg_proj second half): [NH*HD, seq], data [seq, NH*HD].
    # gate is already contiguous (from qg split + reshape), skip explicit contiguous.
    gate_sigmoid = g.sigmoid(gate)
    attn = g.mul(attn, gate_sigmoid)

    # ---- o_proj ----
    w_o = g.weight(os.path.join(weights_dir, f"{pfx}_o_proj_weight.weights"),
                   (hidden_size, num_heads * head_dim), Precision.FP16)
    out = g.matmul(attn, w_o)
    return out


def convert_qwen35(model_dir: str, output_path: str, num_layers: int = 24,
                    prefill_seq_len: int = 256, n_ctx: int = 4096,
                    quant: str = "none"):
    """Main entry point: export weights + build graphs → single .mollm file.

    Args:
        model_dir: path to HF model directory (with config.json + safetensors)
        output_path: output .mollm file path
        num_layers: number of hidden layers
        prefill_seq_len: prefill sequence length
        n_ctx: max context length
    """
    model_dir = Path(model_dir)
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="mollm_weights_")
    weights_dir = tmp_dir
    weights_rel = "."

    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)

    # Find safetensors files (may be sharded)
    st_files = sorted(model_dir.glob('model.safetensors-*.safetensors'))
    if not st_files:
        st_files = list(model_dir.glob('model.safetensors'))
    if not st_files:
        raise FileNotFoundError(f"No safetensors file in {model_dir}")

    # Load all shards and merge
    weights = {}
    for st_path in st_files:
        weights.update(load_safetensors(str(st_path)))

    # ---- Step 1: Export weights to temp dir ----
    print("Exporting weights...")
    export_weights(weights, str(weights_dir), quant=quant)

    # ---- Step 2: Build prefill graph ----
    print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
    g_prefill = build_graph(weights_rel, cfg, seq_len=prefill_seq_len,
                            n_ctx=n_ctx, is_prefill=True)

    # ---- Step 3: Build decode graph ----
    print(f"\nBuilding decode graph (seq_len=1)...")
    g_decode = build_graph(weights_rel, cfg, seq_len=1, n_ctx=n_ctx)

    # ---- Step 4: Pack into single .mollm file ----
    print(f"\nPacking {output_path}...")
    tc = cfg['text_config']
    metadata = {
        "model_name": f"Qwen3.5-{num_layers}L",
        "architecture": "qwen3.5",
        "num_layers": num_layers,
        "hidden_size": tc['hidden_size'],
        "num_heads": tc['num_attention_heads'],
        "num_kv_heads": tc['num_key_value_heads'],
        "head_dim": tc['head_dim'],
        "prefill_seq_len": prefill_seq_len,
        "n_ctx": n_ctx,
        "vocab_size": tc['vocab_size'],
        "layer_types": tc['layer_types'],
        "quantization": quant,
    }
    save_package(output_path, g_prefill, g_decode, weights_dir, metadata,
                 tokenizer_path=str(model_dir / "tokenizer.json"),
                 jinja_path=str(model_dir / "chat_template.jinja"))

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir)
    print(f"\nDone! Output: {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [num_layers] [prefill_seq_len] [quant=none|w8pc|w8g128]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    num_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    prefill_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    quant = sys.argv[5] if len(sys.argv) > 5 else "none"
    convert_qwen35(model_dir, output_path, num_layers, prefill_seq_len, quant=quant)
