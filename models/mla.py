"""
Youtu-LLM MLA model → PROJECT_NAME graph converter.

Usage:
    python models/mla.py /path/to/Youtu-LLM-2B /path/to/output_prefix

Reads config.json + model.safetensors, writes output_prefix.graph.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

# Add parent to path for transpile import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.transpile import GraphBuilder, OpType, Precision, _write_weight_file


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
                arr = arr.astype(np.float32)
            elif dtype_str == 'BF16':
                # BF16 is FP32 with lower 16 bits zeroed
                # uint16 → uint32 shift left 16 → view as float32
                as_u32 = arr.astype(np.uint32) << 16
                arr = as_u32.view(np.float32)
            tensors[name] = arr
    return tensors


def convert_mla(model_dir: str, output_prefix: str, num_layers: int = 32):
    model_dir = Path(model_dir)
    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)

    weights = load_safetensors(str(model_dir / 'model.safetensors'))

    g = GraphBuilder()
    n_layers = num_layers or cfg.get('num_hidden_layers', 32)

    num_heads    = cfg['num_attention_heads']
    kv_lora_rank = cfg['kv_lora_rank']
    q_lora_rank  = cfg['q_lora_rank']
    qk_nope_dim  = cfg['qk_nope_head_dim']
    qk_rope_dim  = cfg['qk_rope_head_dim']
    qk_head_dim  = qk_nope_dim + qk_rope_dim
    v_head_dim   = cfg['v_head_dim']
    hidden_size  = cfg['hidden_size']
    eps          = cfg.get('rms_norm_eps', 1e-6)
    scale        = qk_head_dim ** -0.5
    rope_theta   = cfg.get('rope_theta', 500000.0)

    print(f"MLA: {n_layers} layers, heads={num_heads}, "
          f"qk_nope={qk_nope_dim}, qk_rope={qk_rope_dim}, v={v_head_dim}")

    # ---- write weights ----
    weights_dir = Path(output_prefix).parent or Path('.')
    weight_prefix = Path(output_prefix).name

    def save_weight(name: str, data: np.ndarray):
        wpath = weights_dir / f"{weight_prefix}_{name}.weights"
        _write_weight_file(str(wpath), data)

    for wname, wdata in weights.items():
        # Skip norm weights — they'll be saved by _get_weight with FP32 conversion
        if 'layernorm' in wname.lower() or 'norm' in wname.lower():
            continue
        # Skip embed_tokens — saved separately below
        if 'embed_tokens' in wname:
            continue
        save_weight(wname.replace('.', '_'), wdata)

    # ---- save embed_tokens as FP32 for lookup ----
    embed_w = weights['model.embed_tokens.weight'].astype(np.float32)
    save_weight('embed_tokens', embed_w)
    # Also create a graph node so the engine can find it
    embed_node = g.weight(str(weights_dir / f"{weight_prefix}_embed_tokens.weights"),
                          embed_w.shape, Precision.FP32)

    # ---- graph inputs ----
    hidden = g.input('hidden', (hidden_size,))
    mask   = g.input('mask', (1, 1))        # will be resized at runtime
    cos    = g.input('cos', (qk_rope_dim // 2,))
    sin    = g.input('sin', (qk_rope_dim // 2,))

    # KV cache inputs (preallocated)
    n_ctx = 4096
    cache_inputs = []
    for i in range(n_layers):
        ck = g.input(f'cache_k{i}', (qk_head_dim, n_ctx, num_heads))
        cv = g.input(f'cache_v{i}', (v_head_dim, n_ctx, num_heads))
        cache_inputs.append((ck, cv))

    # ---- build layers ----
    x = hidden
    for i in range(n_layers):
        print(f"  Layer {i}/{n_layers}...", end='\r')
        ck_in, cv_in = cache_inputs[i]
        x, ck_out, cv_out = _build_mla_layer(
            g, x, i, weights, cfg,
            cos, sin, mask, ck_in, cv_in,
            num_heads, kv_lora_rank, q_lora_rank,
            qk_nope_dim, qk_rope_dim, qk_head_dim, v_head_dim,
            hidden_size, eps, scale)
        # replace cache tensors with output views
        cache_inputs[i] = (ck_out, cv_out)

    # ---- final norm ----
    final_norm_w = weights['model.norm.weight']
    save_weight('final_norm', final_norm_w)
    w_norm = g.weight(f'{weight_prefix}_final_norm.weights',
                      (hidden_size,), Precision.FP32)
    x = g.rms_norm(x, w_norm, eps=eps)

    # ---- graph outputs ----
    # x is the final hidden states
    # cache outputs are already linked

    print(f"\n  Total: {len(g._nodes)} nodes")
    g.save(output_prefix)


def _build_mla_layer(g: GraphBuilder, x: int, layer_idx: int,
                     weights: dict, cfg: dict,
                     cos: int, sin: int, mask: int,
                     ck_in: int, cv_in: int,
                     num_heads: int, kv_lora_rank: int, q_lora_rank: int,
                     qk_nope_dim: int, qk_rope_dim: int, qk_head_dim: int,
                     v_head_dim: int, hidden_size: int,
                     eps: float, scale: float
                     ) -> tuple[int, int, int]:
    pfx = f'model.layers.{layer_idx}.self_attn'
    weight_prefix = Path(f'{pfx}').name  # simplified

    # ---- Input RMSNorm (applied before Q and KV projections) ----
    ln_key = f'model.layers.{layer_idx}.input_layernorm.weight'
    w_ln = _get_weight(g, weights, ln_key, f'layer{layer_idx}_input_ln', is_norm=True)
    x_normed = g.rms_norm(x, w_ln, eps=eps)

    # ---- Q path ----
    # q_a_proj: hidden → q_lora_rank
    w_q_a = _get_weight(g, weights, f'{pfx}.q_a_proj.weight', f'layer{layer_idx}_q_a')
    q_comp = g.matmul(x_normed, w_q_a)
    # q_a_layernorm
    w_q_a_norm = _get_weight(g, weights, f'{pfx}.q_a_layernorm.weight', f'layer{layer_idx}_q_a_norm', is_norm=True)
    q_comp = g.rms_norm(q_comp, w_q_a_norm, eps=eps)
    # q_b_proj: q_lora_rank → num_heads * qk_head_dim
    w_q_b = _get_weight(g, weights, f'{pfx}.q_b_proj.weight', f'layer{layer_idx}_q_b')
    q = g.matmul(q_comp, w_q_b)
    # reshape + permute: (num_heads*qk_head_dim, seq) → (qk_head_dim, num_heads, seq)
    q = g.reshape(q, (num_heads * qk_head_dim, -1))
    q = g.reshape(q, (qk_head_dim, num_heads, -1))
    q = g.permute(q, (1, 0, 2, 3))  # → (num_heads, qk_head_dim, seq)
    q = g.reshape(q, (qk_head_dim, -1, num_heads))  # → ncnn format: [hd, seq, heads]
    q_nope, q_rope = g.slice(q, [qk_nope_dim, qk_rope_dim], dim=0)

    # ---- KV path ----
    # kv_a_proj: hidden → kv_lora_rank + qk_rope_dim
    w_kv_a = _get_weight(g, weights, f'{pfx}.kv_a_proj_with_mqa.weight', f'layer{layer_idx}_kv_a')
    kv = g.matmul(x_normed, w_kv_a)
    kv_compressed, k_rope_raw = g.slice(kv, [kv_lora_rank, qk_rope_dim], dim=0)

    # kv_b_proj: kv_lora_rank → num_heads * (qk_nope_dim + v_head_dim)
    w_kv_a_norm = _get_weight(g, weights, f'{pfx}.kv_a_layernorm.weight', f'layer{layer_idx}_kv_a_norm', is_norm=True)
    kv_normed = g.rms_norm(kv_compressed, w_kv_a_norm, eps=eps)
    w_kv_b = _get_weight(g, weights, f'{pfx}.kv_b_proj.weight', f'layer{layer_idx}_kv_b')
    kv_expanded = g.matmul(kv_normed, w_kv_b)
    kv_expanded = g.reshape(kv_expanded, (num_heads * (qk_nope_dim + v_head_dim), -1))
    kv_expanded = g.reshape(kv_expanded, (qk_nope_dim + v_head_dim, num_heads, -1))
    kv_expanded = g.permute(kv_expanded, (1, 0, 2, 3))
    kv_expanded = g.reshape(kv_expanded, (qk_nope_dim + v_head_dim, -1, num_heads))
    k_nope, v = g.slice(kv_expanded, [qk_nope_dim, v_head_dim], dim=0)

    # ---- RoPE ----
    q_rope = g.rope(q_rope, cos, sin, rope_dim=qk_rope_dim)
    k_rope_raw = g.reshape(k_rope_raw, (qk_rope_dim, -1, 1))
    k_rope = g.rope(k_rope_raw, cos, sin, rope_dim=qk_rope_dim)
    k_rope = g.tile(k_rope, (num_heads, 1, 1))  # broadcast to all heads

    # ---- concat ----
    q_full = g.concat([q_nope, q_rope], dim=0)
    k_full = g.concat([k_nope, k_rope], dim=0)

    # ---- SDPA ----
    attn_out, ck_out, cv_out = g.sdpa(
        q_full, k_full, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=scale,
        num_heads=num_heads, num_kv_heads=num_heads,
        head_dim=qk_head_dim, v_head_dim=v_head_dim)

    # ---- output projection ----
    # attn_out: [v_head_dim, seq, num_heads]
    attn_out = g.permute(attn_out, (1, 0, 2, 3))
    attn_out = g.reshape(attn_out, (num_heads * v_head_dim, -1))
    w_o = _get_weight(g, weights, f'{pfx}.o_proj.weight', f'layer{layer_idx}_o')
    out = g.matmul(attn_out, w_o)
    x = g.add(x, out)

    # ---- MLP (SwiGLU) ----
    # post_attention_layernorm -> gate_proj + up_proj -> SiLU(gate) * up -> down_proj
    mlp_pfx = f'model.layers.{layer_idx}.mlp'
    ln2_key = f'model.layers.{layer_idx}.post_attention_layernorm.weight'
    w_ln2 = _get_weight(g, weights, ln2_key, f'layer{layer_idx}_post_ln', is_norm=True)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    w_gate = _get_weight(g, weights, f'{mlp_pfx}.gate_proj.weight', f'layer{layer_idx}_gate')
    w_up   = _get_weight(g, weights, f'{mlp_pfx}.up_proj.weight',   f'layer{layer_idx}_up')
    w_down = _get_weight(g, weights, f'{mlp_pfx}.down_proj.weight', f'layer{layer_idx}_down')

    gate = g.matmul(x_normed2, w_gate)
    up   = g.matmul(x_normed2, w_up)
    gate = g.silu(gate)
    mlp_hidden = g.mul(gate, up)
    mlp_out = g.matmul(mlp_hidden, w_down)
    x = g.add(x, mlp_out)

    return x, ck_out, cv_out


def _get_weight(g: GraphBuilder, weights: dict, key: str,
                save_name: str, is_norm: bool = False) -> int:
    """Load weight from safetensors dict, save as .weights file, return node id."""
    data = weights[key]
    # Convert BF16/FP16 to FP32 for ALL weights (engine always uses FP32)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    # Transpose to [K, N] = [in_features, out_features] for matmul convention
    # Safetensors stores [out_features, in_features], we need [in_features, out_features]
    data = data.T.copy()
    prec = Precision.FP32
    shape = tuple(data.shape)
    weight_prefix = Path(save_name).name
    weights_dir = Path('.')
    wpath = weights_dir / f"{weight_prefix}.weights"
    _write_weight_file(str(wpath), data)
    return g.weight(str(wpath), shape, prec)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output_prefix> [num_layers]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_prefix = sys.argv[2]
    num_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    convert_mla(model_dir, output_prefix, num_layers)
