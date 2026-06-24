"""
Youtu-LLM MLA model → mlllm graph converter.

Usage:
    python models/mla.py /path/to/Youtu-LLM-2B /path/to/output_dir

Outputs:
    output_dir/
    ├── weights/               # shared weight files (written once)
    │   ├── embed_tokens.weights
    │   ├── layer0_input_ln.weights
    │   └── ...
    ├── model_prefill.graph    # prefill graph (seq_len=128)
    └── model_decode.graph     # decode graph (seq_len=1)

Reads config.json + model.safetensors.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

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
                arr = arr.astype(np.float16)  # keep FP16
            elif dtype_str == 'BF16':
                # BF16 → FP32 (no native FP16 BF16 support)
                as_u32 = arr.astype(np.uint32) << 16
                arr = as_u32.view(np.float32)
            tensors[name] = arr
    return tensors


def export_weights(weights: dict, weights_dir: str):
    """Export all weights to a shared directory. Called once."""
    os.makedirs(weights_dir, exist_ok=True)

    def save(name: str, data: np.ndarray):
        wpath = os.path.join(weights_dir, f"{name}.weights")
        _write_weight_file(wpath, data)

    # All projection weights (non-norm) — keep as FP16.
    for wname, wdata in weights.items():
        if 'layernorm' in wname.lower() or 'norm' in wname.lower():
            continue
        if 'embed_tokens' in wname:
            continue
        # Convert to float16 for FP16 storage.
        d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
        save(wname.replace('.', '_'), d)

    # embed_tokens — FP16, packed at load time (tied with lm_head)
    embed_w = weights['model.embed_tokens.weight']
    d = embed_w.astype(np.float16) if embed_w.dtype != np.float16 else embed_w
    save('embed_tokens', d)

    # norm weights (FP32)
    for wname, wdata in weights.items():
        if 'norm' in wname.lower() or 'layernorm' in wname.lower():
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            save(wname.replace('.', '_'), d)

    # final norm
    final_norm_w = weights['model.norm.weight']
    d = final_norm_w.astype(np.float32) if final_norm_w.dtype != np.float32 else final_norm_w
    save('final_norm', d)


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096) -> GraphBuilder:
    """
    Build a computation graph with the given seq_len.
    seq_len=1 for decode, seq_len=128 for prefill chunk.
    """
    g = GraphBuilder()
    n_layers = cfg.get('num_hidden_layers', 32)

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

    print(f"MLA graph: seq_len={seq_len}, heads={num_heads}, "
          f"qk_nope={qk_nope_dim}, qk_rope={qk_rope_dim}, v={v_head_dim}")

    # ---- embed_tokens weight node ----
    embed_path = os.path.join(weights_dir, "embed_tokens.weights")
    embed_shape = (cfg['vocab_size'], hidden_size)
    embed_node = g.weight(embed_path, embed_shape, Precision.FP16)

    # ---- graph inputs ----
    hidden = g.input('hidden', (hidden_size, seq_len))
    mask   = g.input('mask', (1, seq_len))
    cos    = g.input('cos', (qk_rope_dim // 2, seq_len))
    sin    = g.input('sin', (qk_rope_dim // 2, seq_len))

    # KV cache inputs (preallocated, metadata in header)
    cache_inputs = []
    for i in range(n_layers):
        ck = g.input(f'cache_k{i}', (qk_head_dim, n_ctx, num_heads))
        cv = g.input(f'cache_v{i}', (v_head_dim, n_ctx, num_heads))
        cache_inputs.append((ck, cv))

    # ---- build layers ----
    x = hidden
    for i in range(n_layers):
        ck_in, cv_in = cache_inputs[i]
        x, ck_out, cv_out = _build_mla_layer(
            g, x, i, weights_dir, cfg,
            cos, sin, mask, ck_in, cv_in,
            num_heads, kv_lora_rank, q_lora_rank,
            qk_nope_dim, qk_rope_dim, qk_head_dim, v_head_dim,
            hidden_size, eps, scale, seq_len)
        cache_inputs[i] = (ck_out, cv_out)

    # ---- final norm ----
    w_norm = g.weight(os.path.join(weights_dir, "final_norm.weights"),
                      (hidden_size,), Precision.FP32)
    x = g.rms_norm(x, w_norm, eps=eps)

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def _build_mla_layer(g: GraphBuilder, x: int, layer_idx: int,
                     weights_dir: str, cfg: dict,
                     cos: int, sin: int, mask: int,
                     ck_in: int, cv_in: int,
                     num_heads: int, kv_lora_rank: int, q_lora_rank: int,
                     qk_nope_dim: int, qk_rope_dim: int, qk_head_dim: int,
                     v_head_dim: int, hidden_size: int,
                     eps: float, scale: float, seq_len: int
                     ) -> tuple[int, int, int]:
    pfx = f'model.layers.{layer_idx}.self_attn'

    # ---- Input RMSNorm ----
    w_ln = g.weight(os.path.join(weights_dir, f"model_layers_{layer_idx}_input_layernorm_weight.weights"),
                    (hidden_size,), Precision.FP32)
    x_normed = g.rms_norm(x, w_ln, eps=eps)

    # ---- Q path ----
    w_q_a = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_q_a_proj_weight.weights"),
                     (q_lora_rank, hidden_size), Precision.FP16)
    q_comp = g.matmul(x_normed, w_q_a)

    w_q_a_norm = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_q_a_layernorm_weight.weights"),
                          (q_lora_rank,), Precision.FP32)
    q_comp = g.rms_norm(q_comp, w_q_a_norm, eps=eps)

    w_q_b = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_q_b_proj_weight.weights"),
                     (num_heads * qk_head_dim, q_lora_rank), Precision.FP16)
    q = g.matmul(q_comp, w_q_b)

    # reshape + permute: (num_heads*qk_head_dim, seq) → [hd, seq, heads]
    q = g.reshape(q, (num_heads * qk_head_dim, seq_len))
    q = g.reshape(q, (qk_head_dim, num_heads, seq_len))
    q = g.permute(q, (0, 2, 1, 3))  # [hd, seq, heads]
    q_nope, q_rope = g.slice(q, [qk_nope_dim, qk_rope_dim], dim=0)

    # ---- KV path ----
    w_kv_a = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_kv_a_proj_with_mqa_weight.weights"),
                      (kv_lora_rank + qk_rope_dim, hidden_size), Precision.FP16)
    kv = g.matmul(x_normed, w_kv_a)
    kv_compressed, k_rope_raw = g.slice(kv, [kv_lora_rank, qk_rope_dim], dim=0)

    w_kv_a_norm = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_kv_a_layernorm_weight.weights"),
                           (kv_lora_rank,), Precision.FP32)
    kv_normed = g.rms_norm(kv_compressed, w_kv_a_norm, eps=eps)

    w_kv_b = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_kv_b_proj_weight.weights"),
                      (num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank), Precision.FP16)
    kv_expanded = g.matmul(kv_normed, w_kv_b)
    kv_expanded = g.reshape(kv_expanded, (num_heads * (qk_nope_dim + v_head_dim), seq_len))
    kv_expanded = g.reshape(kv_expanded, (qk_nope_dim + v_head_dim, num_heads, seq_len))
    kv_expanded = g.permute(kv_expanded, (0, 2, 1, 3))  # [qk_nope+v, seq, heads]
    k_nope, v = g.slice(kv_expanded, [qk_nope_dim, v_head_dim], dim=0)

    # ---- RoPE ----
    q_rope = g.rope(q_rope, cos, sin, rope_dim=qk_rope_dim)
    k_rope_raw = g.reshape(k_rope_raw, (qk_rope_dim, seq_len, 1))
    k_rope = g.rope(k_rope_raw, cos, sin, rope_dim=qk_rope_dim)
    k_rope = g.tile(k_rope, (1, 1, num_heads))  # broadcast to all heads along dim=2

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
    attn_out = g.permute(attn_out, (0, 2, 1, 3))  # [v, heads, seq]
    attn_out = g.reshape(attn_out, (num_heads * v_head_dim, seq_len))
    w_o = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_o_proj_weight.weights"),
                   (hidden_size, num_heads * v_head_dim), Precision.FP16)
    out = g.matmul(attn_out, w_o)
    x = g.add(x, out)

    # ---- MLP (SwiGLU) ----
    mlp_pfx = f'model.layers.{layer_idx}.mlp'

    w_ln2 = g.weight(os.path.join(weights_dir, f"model_layers_{layer_idx}_post_attention_layernorm_weight.weights"),
                     (hidden_size,), Precision.FP32)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    w_gate = g.weight(os.path.join(weights_dir, f"{mlp_pfx.replace('.', '_')}_gate_proj_weight.weights"),
                      (cfg_intermediate_size(cfg), hidden_size), Precision.FP16)
    w_up   = g.weight(os.path.join(weights_dir, f"{mlp_pfx.replace('.', '_')}_up_proj_weight.weights"),
                      (cfg_intermediate_size(cfg), hidden_size), Precision.FP16)
    w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx.replace('.', '_')}_down_proj_weight.weights"),
                      (hidden_size, cfg_intermediate_size(cfg)), Precision.FP16)

    gate = g.matmul(x_normed2, w_gate)
    up   = g.matmul(x_normed2, w_up)
    gate = g.silu(gate)
    mlp_hidden = g.mul(gate, up)
    mlp_out = g.matmul(mlp_hidden, w_down)
    x = g.add(x, mlp_out)

    return x, ck_out, cv_out


def cfg_intermediate_size(cfg: dict) -> int:
    return cfg.get('intermediate_size', cfg['hidden_size'] * 3)


def convert_mla(model_dir: str, output_dir: str, num_layers: int = 32,
                prefill_seq_len: int = 128, n_ctx: int = 4096):
    """Main entry point: export weights + build prefill/decode graphs."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    weights_dir = output_dir / "weights"
    # Graph stores paths relative to the graph file's directory.
    # The graph is saved in output_dir/, so weights paths should be "weights/...".
    weights_rel = "weights"

    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)

    weights = load_safetensors(str(model_dir / 'model.safetensors'))

    # ---- Step 1: Export shared weights ----
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
    print(f"  {output_dir}/weights/  — shared weight files")
    print(f"  {output_dir}/model_prefill.graph")
    print(f"  {output_dir}/model_decode.graph")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output_dir> [num_layers] [prefill_seq_len]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    prefill_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    convert_mla(model_dir, output_dir, num_layers, prefill_seq_len)
