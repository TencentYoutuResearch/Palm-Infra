"""
mlllm model builder for Qwen3.5-0.8B (text only).

Hybrid architecture:
  - 18 layers of linear attention (Gated Delta Rule)
  - 6 layers of full attention (GQA + QK norm + output gate)
  - Standard SwiGLU MLP

Text-only: vision encoder weights are ignored.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

import numpy as np

from python.transpile import GraphBuilder, OpType, Precision, Activation, _write_weight_file


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


def export_weights(weights: dict, weights_dir: str):
    """Export text model weights. Skip vision encoder."""
    os.makedirs(weights_dir, exist_ok=True)

    def save(name: str, data: np.ndarray):
        wpath = os.path.join(weights_dir, f"{name}.weights")
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
            save('final_norm', d)
            continue

        # Norms → FP32
        if 'norm' in wname.lower() or 'layernorm' in wname.lower():
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
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

        # All projection weights → FP16
        d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
        save(wname.replace('.', '_'), d)


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096) -> GraphBuilder:
    """Build prefill or decode graph for Qwen3.5 text model."""
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
    conv_kernel = tc['linear_conv_kernel_dim']       # 4

    # MLP
    intermediate = tc.get('intermediate_size', 3584)

    print(f"Qwen3.5 graph: seq_len={seq_len}, layers={num_layers}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"linear_heads={linear_num_heads}, linear_k={linear_k_dim}, "
          f"linear_v={linear_v_dim}, conv_kernel={conv_kernel}")

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
    hidden = g.input('hidden', (hidden_size, seq_len))
    mask = g.input('mask', (1, seq_len))
    cos = g.input('cos', (rope_dim // 2, seq_len))
    sin = g.input('sin', (rope_dim // 2, seq_len))

    # ---- persistent state inputs (KV cache for full attn, GDN state for linear attn) ----
    cache_inputs = []
    for i in range(num_layers):
        lt = layer_types[i]
        if lt == 'full_attention':
            ck = g.input(f'cache_k{i}', (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cv = g.input(f'cache_v{i}', (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cache_inputs.append(('kv', ck, cv))
        else:
            # GDN recurrent state: [v_dim, k_dim, num_heads] FP32
            gs = g.input(f'gdn_state{i}', (linear_v_dim, linear_k_dim, linear_num_heads), prec=Precision.FP32)
            # Conv state: [qkv_dim, conv_kernel-1] FP16
            qkv_dim = linear_num_heads * linear_k_dim * 3  # 16 * 128 * 3 = 6144
            gc = g.input(f'gdn_conv{i}', (qkv_dim, conv_kernel - 1), prec=Precision.FP16)
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
                        conv_kernel, intermediate, hidden_size, lt)

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
                 conv_kernel, intermediate, hidden_size, layer_type):
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
                                             conv_kernel, hidden_size)
    else:
        attn_out = _build_full_attn_layer(g, x_normed, layer_idx, weights_dir,
                                           cos, sin, mask, ck_in, cv_in,
                                           eps, seq_len, rope_dim,
                                           num_heads, num_kv_heads, head_dim,
                                           hidden_size)

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
                              num_heads, k_dim, v_dim, conv_kernel, hidden_size):
    """Build a linear attention (Gated Delta Rule) layer."""
    pfx = f'model_language_model_layers_{layer_idx}_linear_attn'

    # ---- Projections ----
    w_qkv = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_qkv_weight.weights"),
                     (num_heads * k_dim * 3, hidden_size), Precision.FP16)
    qkv = g.matmul(x, w_qkv)  # [16*128*3, seq]

    w_a = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_a_weight.weights"),
                   (num_heads, hidden_size), Precision.FP16)
    a_out = g.matmul(x, w_a)  # [16, seq]

    w_b = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_b_weight.weights"),
                   (num_heads, hidden_size), Precision.FP16)
    b_out = g.matmul(x, w_b)  # [16, seq]

    w_z = g.weight(os.path.join(weights_dir, f"{pfx}_in_proj_z_weight.weights"),
                   (num_heads * v_dim, hidden_size), Precision.FP16)
    z_out = g.matmul(x, w_z)  # [16*128, seq]

    # ---- A_log and dt_bias (per-head constants) ----
    A_log = g.weight(os.path.join(weights_dir, f"{pfx}_A_log.weights"),
                     (num_heads,), Precision.FP32)
    dt_bias = g.weight(os.path.join(weights_dir, f"{pfx}_dt_bias.weights"),
                       (num_heads,), Precision.FP32)

    # ---- Compute g = -exp(A_log) * softplus(a + dt_bias) ----
    # A_log is [16], dt_bias is [16]. Need to broadcast to [16, seq].
    # a_out is [16, seq]. We need a + dt_bias (broadcast dt_bias across seq).
    # Use scalar_add per-head? No — dt_bias is per-head, a_out is [16, seq].
    # Since A_log/dt_bias are CONSTANT nodes (shape [16, 1, 1, 1]),
    # and a_out is [16, seq, 1, 1], ADD will broadcast.
    # But ADD broadcasts by taking max of each dim — [16,1,1,1] + [16,seq,1,1] → [16,seq,1,1]. OK.
    a_plus_dt = g.add(a_out, dt_bias)
    sp = g.softplus(a_plus_dt)          # softplus(a + dt_bias)
    neg_exp_A = g.scalar_mul(g.exp(A_log), -1.0)  # -exp(A_log)
    g_val = g.mul(neg_exp_A, sp)        # g = -exp(A_log) * softplus(a + dt_bias)

    # ---- Compute beta = sigmoid(b) ----
    beta_val = g.sigmoid(b_out)

    # ---- Short conv on qkv (conv1d + silu) ----
    # matmul output Tensor shape is [qkv_total, seq] but data is [seq, qkv_total]
    # (transposed). ShortConv expects [groups, seq] = [qkv_total, seq] data.
    # Reshape to [seq, qkv_total] (matching data) then permute to [qkv_total, seq].
    qkv_fixed = g.reshape(qkv, (seq_len, num_heads * k_dim * 3))  # materialize
    qkv_fixed = g.permute(qkv_fixed, (1, 0, 2, 3))  # [qkv_total, seq, 1, 1]
    # conv1d.weight: [6144, 1, 4] → reshape to [6144, 4] (groups=6144, kernel_size=4)
    w_conv = g.weight(os.path.join(weights_dir, f"{pfx}_conv1d_weight.weights"),
                     (num_heads * k_dim * 3, conv_kernel), Precision.FP32)
    qkv_conv = g.shortconv(qkv_fixed, w_conv, gc_in, kernel_size=conv_kernel)

    # ---- Split qkv_conv into q, k, v ----
    # matmul output Tensor shape is [qkv_total, seq] but data is [seq, qkv_total]
    # row-major (transposed). shortconv preserves this layout.
    # Need to reshape to [seq, qkv_total] (matching data), then permute to
    # [qkv_total, seq], then split into q/k/v.
    # Actually: reshape [qkv_total, seq] → materialize to [seq, qkv_total] by
    # reshaping to (seq_len, qkv_total) which triggers a copy (non-contiguous).
    qkv_t = g.reshape(qkv_conv, (seq_len, num_heads * k_dim * 3))  # materialize [seq, qkv_total]
    qkv_t = g.permute(qkv_t, (1, 0, 2, 3))  # [qkv_total, seq, 1, 1]
    # Now split on dim=0: q=[0:2048], k=[2048:4096], v=[4096:6144]
    qkv_dim = num_heads * k_dim  # 2048
    q, k, v = g.slice(qkv_t, [qkv_dim, qkv_dim, qkv_dim], dim=0)
    # q/k: [qkv_dim, seq] = [2048, seq], reshape to [head, k_dim, seq] → permute [head, seq, k_dim]
    q = g.reshape(q, (num_heads, k_dim, seq_len))
    q = g.permute(q, (0, 2, 1, 3))  # [head, seq, k_dim]
    k = g.reshape(k, (num_heads, k_dim, seq_len))
    k = g.permute(k, (0, 2, 1, 3))
    v = g.reshape(v, (num_heads, v_dim, seq_len))
    v = g.permute(v, (0, 2, 1, 3))  # [head, seq, v_dim]

    # ---- GDN op ----
    # g_val is [num_heads, seq] (shape [16, seq, 1, 1] after add/softplus)
    # beta_val is [num_heads, seq]
    # g_val is [num_heads, seq, 1, 1], beta_val is [num_heads, seq, 1, 1].
    # GDN kernel expects [num_heads, seq] (reads g_data + h * seq_len).
    # Just reshape — no permute needed.
    g_val = g.reshape(g_val, (num_heads, seq_len))
    beta_val = g.reshape(beta_val, (num_heads, seq_len))

    scale = k_dim ** -0.5
    gdn_op = OpType.GATED_DELTANET_DECODE if seq_len == 1 else OpType.GATED_DELTANET_PREFILL
    gdn_out = g._add(gdn_op,
                     [q, k, v, g_val, beta_val, gs_in],
                     (v_dim, seq_len, num_heads),  # kernel writes [head, seq, v_dim] but Tensor shape is [v_dim, seq, heads]
                     prec=Precision.FP32,
                     i32=[num_heads, k_dim, v_dim, 1])  # use_qk_l2norm=1

    # ---- RMSNormGated: norm(gdn_out) * silu(z) ----
    # HF: core_attn_out = self.norm(core_attn_out, z)
    #     = rms_norm(core_attn_out, norm.weight) * silu(z)
    # Kernel output layout: [head, seq, v_dim] in memory.
    # Tensor shape declared as (v_dim, seq, num_heads) but actual data is [head, seq, v_dim].
    # Need to reshape to [v_dim, num_heads*seq] for per-head RMSNorm.
    # But reshape preserves flat order: [head, seq, v_dim] flat = head0_seq0_v0..v127, head0_seq1_v0.., head1_seq0_v0..
    # = [head, seq*v_dim] if we think of it as [head, seq*v_dim]
    # For rms_norm on v_dim: need [v_dim, num_heads*seq] where each column is (head, seq) combo.
    # [head, seq, v_dim] → permute to [v_dim, seq, head] → reshape to [v_dim, seq*head]
    gdn_out = g.permute(gdn_out, (2, 1, 0, 3))  # [v_dim, seq, head, 1]
    w_norm = g.weight(os.path.join(weights_dir, f"{pfx}_norm_weight.weights"),
                     (v_dim,), Precision.FP32)
    gdn_out = g.reshape(gdn_out, (v_dim, num_heads * seq_len))
    gdn_out = g.rms_norm(gdn_out, w_norm, eps=eps)
    gdn_out = g.reshape(gdn_out, (num_heads * v_dim, seq_len))
    z_silu = g.silu(z_out)
    gated = g.mul(gdn_out, z_silu)

    # ---- out_proj ----
    w_out = g.weight(os.path.join(weights_dir, f"{pfx}_out_proj_weight.weights"),
                     (hidden_size, num_heads * v_dim), Precision.FP16)
    out = g.matmul(gated, w_out)
    return out


def _build_full_attn_layer(g, x, layer_idx, weights_dir,
                            cos, sin, mask, ck_in, cv_in,
                            eps, seq_len, rope_dim,
                            num_heads, num_kv_heads, head_dim, hidden_size):
    """Build a full attention (GQA + QK norm + output gate) layer."""
    pfx = f'model_language_model_layers_{layer_idx}_self_attn'

    # ---- q_proj: [4096, 1024] → 8 × (256 query + 256 gate) ----
    # q_proj output is chunked: first half = query, second half = gate
    w_q = g.weight(os.path.join(weights_dir, f"{pfx}_q_proj_weight.weights"),
                   (num_heads * head_dim * 2, hidden_size), Precision.FP16)
    qg = g.matmul(x, w_q)  # [8 × 512, seq]
    # Split: query [8 × 256], gate [8 × 256]
    query, gate = g.slice(qg, [num_heads * head_dim, num_heads * head_dim], dim=0)

    # ---- k_proj, v_proj ----
    w_k = g.weight(os.path.join(weights_dir, f"{pfx}_k_proj_weight.weights"),
                   (num_kv_heads * head_dim, hidden_size), Precision.FP16)
    w_v = g.weight(os.path.join(weights_dir, f"{pfx}_v_proj_weight.weights"),
                   (num_kv_heads * head_dim, hidden_size), Precision.FP16)
    k = g.matmul(x, w_k)
    v = g.matmul(x, w_v)

    # ---- QK norm (RMSNorm per head) ----
    # q_norm/k_norm weight shape: [head_dim]
    # query before reshape: [num_heads*head_dim, seq] → reshape to [head_dim, num_heads*seq]
    # RMSNorm operates on dim0 (head_dim), treating each column as a separate vector.
    w_qn = g.weight(os.path.join(weights_dir, f"{pfx}_q_norm_weight.weights"),
                    (head_dim,), Precision.FP32)
    w_kn = g.weight(os.path.join(weights_dir, f"{pfx}_k_norm_weight.weights"),
                    (head_dim,), Precision.FP32)
    query = g.reshape(query, (head_dim, num_heads * seq_len))
    query = g.rms_norm(query, w_qn, eps=eps)
    k = g.reshape(k, (head_dim, num_kv_heads * seq_len))
    k = g.rms_norm(k, w_kn, eps=eps)

    # ---- Reshape for multi-head: [head_dim, seq, num_heads] ----
    query = g.reshape(query, (head_dim, seq_len, num_heads))
    k = g.reshape(k, (head_dim, seq_len, num_kv_heads))
    v = g.reshape(v, (head_dim, seq_len, num_kv_heads))

    # ---- RoPE (partial: only rope_dim=64 out of head_dim=256) ----
    # kernel_rope applies rotation to [0, rope_dim) and copies [rope_dim, D) unchanged.
    query = g.rope(query, cos, sin, rope_dim=rope_dim, interleave=True)
    k = g.rope(k, cos, sin, rope_dim=rope_dim, interleave=True)

    # ---- SDPA ----
    scale = head_dim ** -0.5
    attn, ck_out, cv_out = g.sdpa(
        query, k, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=scale,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, v_head_dim=head_dim)

    # ---- Reshape attn from [head_dim, seq, num_heads] → [num_heads*head_dim, seq] ----
    attn = g.reshape(attn, (num_heads * head_dim, seq_len))

    # ---- Output gate: attn * sigmoid(gate) ----
    # gate shape: [num_heads*head_dim, seq] — same as reshaped attn
    gate_sigmoid = g.sigmoid(gate)
    attn = g.mul(attn, gate_sigmoid)

    # ---- o_proj ----
    w_o = g.weight(os.path.join(weights_dir, f"{pfx}_o_proj_weight.weights"),
                   (hidden_size, num_heads * head_dim), Precision.FP16)
    out = g.matmul(attn, w_o)
    return out


def convert_qwen35(model_dir: str, output_dir: str, num_layers: int = 24,
                    prefill_seq_len: int = 256, n_ctx: int = 4096):
    """Main entry point: export weights + build prefill/decode graphs."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    weights_dir = output_dir / "weights"
    weights_rel = "weights"

    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)

    # Find safetensors file
    st_files = list(model_dir.glob('model.safetensors-*.safetensors'))
    if not st_files:
        st_files = list(model_dir.glob('model.safetensors'))
    if not st_files:
        raise FileNotFoundError(f"No safetensors file in {model_dir}")

    weights = load_safetensors(str(st_files[0]))

    # ---- Step 1: Export weights ----
    print("Exporting weights...")
    export_weights(weights, str(weights_dir))

    # ---- Step 2: Build prefill graph ----
    print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
    g_prefill = build_graph(weights_rel, cfg, seq_len=prefill_seq_len, n_ctx=n_ctx)
    g_prefill.save(str(output_dir / "model_prefill"))

    # ---- Step 3: Build decode graph ----
    print(f"\nBuilding decode graph (seq_len=1)...")
    g_decode = build_graph(weights_rel, cfg, seq_len=1, n_ctx=n_ctx)
    g_decode.save(str(output_dir / "model_decode"))

    print(f"\nDone! Output in {output_dir}/")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output_dir> [num_layers] [prefill_seq_len]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    prefill_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    convert_qwen35(model_dir, output_dir, num_layers, prefill_seq_len)
