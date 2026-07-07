"""
Youtu-LLM MLA model → mollm graph converter.

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
from transpile import (
    GraphBuilder, OpType, Precision, Activation, _write_weight_file,
    quantize_weight_w8_group, save_package,
    write_quantized_weight_file_cpp,
)


def _w4mix_promote_to_w8(wname: str) -> bool:
    """Youtu MLA mixed W4 policy.

    Keep output/expansion-heavy tensors in W8 and leave compression/gate-up
    tensors in W4. This is deliberately conservative enough to protect quality
    without turning the package into mostly-W8.
    """
    if wname == "lm_head.weight" or wname.endswith(".lm_head.weight"):
        return True
    if ".self_attn.kv_b_proj.weight" in wname:
        return True
    if ".self_attn.o_proj.weight" in wname:
        return True
    if ".mlp.down_proj.weight" in wname:
        return True
    return False


def _canonical_quant(quant: str) -> str:
    q = quant.lower()
    if q in ("none", "fp16", "f16"):
        return "fp16"
    return q


def _quant_spec(quant: str, k: int, wname: str = "") -> tuple[str, int] | None:
    quant = _canonical_quant(quant)
    if quant == "fp16":
        return None
    if quant == "w8pc":
        return ("w8", k)
    if quant.startswith("w8g"):
        return ("w8", int(quant[3:]))
    if quant.startswith("w4mixg"):
        group_size = int(quant[6:])
        if _w4mix_promote_to_w8(wname):
            return ("w8", group_size)
        return ("w4", group_size)
    if quant.startswith("w4g"):
        return ("w4", int(quant[3:]))
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
                arr = arr.astype(np.float16)  # keep FP16
            elif dtype_str == 'BF16':
                # BF16 → FP32 (no native FP16 BF16 support)
                as_u32 = arr.astype(np.uint32) << 16
                arr = as_u32.view(np.float32)
            tensors[name] = arr
    return tensors


def export_weights(weights: dict, weights_dir: str, quant: str = "fp16"):
    """Export all weights to a shared directory. Called once."""
    os.makedirs(weights_dir, exist_ok=True)
    quant = _canonical_quant(quant)
    quant_counts = {"w4": 0, "w8": 0}
    lm_head_source = weights.get('lm_head.weight')
    if lm_head_source is None:
        lm_head_source = weights.get('model.lm_head.weight')

    def save(name: str, data: np.ndarray, quantizable: bool = False, raw_name: str = ""):
        wpath = os.path.join(weights_dir, f"{name}.weights")
        quant_spec = (
            _quant_spec(quant, data.shape[1], raw_name or name)
            if quantizable and data.ndim == 2 else None
        )
        if quant_spec is not None:
            quant_kind, group_size = quant_spec
            if write_quantized_weight_file_cpp(
                wpath, data, quant_kind, group_size, required=(quant_kind == "w4")
            ):
                quant_counts[quant_kind] += 1
                return
            if quant_kind == "w8":
                q, scales, gs, ng = quantize_weight_w8_group(data, group_size)
                _write_weight_file(wpath, q, scales=scales, group_size=gs, num_groups=ng)
            else:
                raise ValueError(f"unsupported quant kind: {quant_kind}")
            quant_counts[quant_kind] += 1
        else:
            _write_weight_file(wpath, data)

    # All projection weights (non-norm) — optionally W8 quantized.
    for wname, wdata in weights.items():
        if 'layernorm' in wname.lower() or 'norm' in wname.lower():
            continue
        if 'embed_tokens' in wname:
            continue
        if wname.endswith('lm_head.weight'):
            continue
        # Skip mlp.gate_proj / mlp.up_proj — they will be merged below.
        if '.mlp.gate_proj.' in wname or '.mlp.up_proj.' in wname:
            continue
        # Convert to float16 for FP16 storage.
        d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
        save(wname.replace('.', '_'), d, quantizable=True, raw_name=wname)

    # Merge mlp.gate_proj + mlp.up_proj per layer into a single weight
    # (one matmul + slice at inference, halves A-pack overhead).
    # Both shapes are [intermediate, hidden]; concat along dim=0 (N) → [2*intermediate, hidden].
    import re
    layer_pattern = re.compile(r'^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$')
    layer_ids = set()
    for wname in weights.keys():
        m = layer_pattern.match(wname)
        if m:
            layer_ids.add(int(m.group(1)))
    for layer_idx in sorted(layer_ids):
        gate_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        up_name   = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        if up_name not in weights:
            continue
        gate_w = weights[gate_name]
        up_w   = weights[up_name]
        merged = np.concatenate([gate_w, up_w], axis=0)
        if merged.dtype != np.float16:
            merged = merged.astype(np.float16)
        save(
            f"model_layers_{layer_idx}_mlp_gate_up_proj_weight",
            merged,
            quantizable=True,
            raw_name=f"model.layers.{layer_idx}.mlp.gate_up_proj.weight",
        )

    # embed_tokens — FP16 row-major for lookup.
    embed_w = weights['model.embed_tokens.weight']
    d = embed_w.astype(np.float16) if embed_w.dtype != np.float16 else embed_w
    save('embed_tokens', d)

    # lm_head is stored explicitly even when the source model ties it to
    # embeddings. Treat it as a normal linear weight for quantization/runtime.
    if lm_head_source is None:
        lm_head_source = embed_w
    d = lm_head_source.astype(np.float16) if lm_head_source.dtype != np.float16 else lm_head_source
    save('lm_head', d, quantizable=True, raw_name="lm_head.weight")

    # norm weights (FP32)
    for wname, wdata in weights.items():
        if 'norm' in wname.lower() or 'layernorm' in wname.lower():
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            save(wname.replace('.', '_'), d)

    # final norm
    final_norm_w = weights['model.norm.weight']
    d = final_norm_w.astype(np.float32) if final_norm_w.dtype != np.float32 else final_norm_w
    save('final_norm', d)

    if quant != "fp16":
        print(f"  Quantized tensors: W4={quant_counts['w4']} W8={quant_counts['w8']}")


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096, is_prefill: bool = False) -> GraphBuilder:
    """
    Build a computation graph with the given seq_len.
    seq_len=1 for decode, seq_len=128 for prefill chunk.

    When is_prefill=True, the hidden INPUT's seq dim is marked DynamicKind.SEQ
    so the C++ runtime can inject actual seq_len (dynamic shape mode).
    Decode graphs (seq=1) stay all-STATIC.
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

    # ---- set graph metadata (engine reads these instead of CLI args) ----
    g.set_model_config(
        rope_dim=qk_rope_dim,
        rope_theta=cfg.get('rope_theta', 1600000),
        hidden_size=hidden_size,
        num_layers=n_layers,
        vocab_size=cfg['vocab_size'],
        model_type='mla',
    )

    # ---- embed_tokens weight node ----
    embed_path = os.path.join(weights_dir, "embed_tokens.weights")
    embed_shape = (cfg['vocab_size'], hidden_size)
    embed_node = g.weight(embed_path, embed_shape, Precision.FP16)

    # ---- lm_head weight node ----
    lm_head_path = os.path.join(weights_dir, "lm_head.weights")
    g.weight(lm_head_path, embed_shape, Precision.FP16)

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
    mask   = g.input('mask',   (1, seq_len),           dynamic=mask_dyn)
    cos    = g.input('cos',    (qk_rope_dim // 2, seq_len), dynamic=cos_dyn)
    sin    = g.input('sin',    (qk_rope_dim // 2, seq_len), dynamic=sin_dyn)

    # KV cache inputs (preallocated, metadata in header)
    # FP16 storage: halves cache memory, enables FP16FML SDPA without per-call
    # FP32→FP16 conversion of the entire KV cache.
    cache_inputs = []
    for i in range(n_layers):
        ck = g.input(f'cache_k{i}', (qk_head_dim, n_ctx, num_heads), prec=Precision.FP16)
        cv = g.input(f'cache_v{i}', (v_head_dim, n_ctx, num_heads), prec=Precision.FP16)
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
            hidden_size, eps, scale, seq_len, is_prefill=is_prefill)
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
                     eps: float, scale: float, seq_len: int,
                     is_prefill: bool = False
                     ) -> tuple[int, int, int]:
    # In prefill graphs, use SEQ symbol for seq dims (runtime substitutes
    # actual seq_len). In decode graphs, use seq_len literal (=1, static).
    from transpile import SEQ
    _S = SEQ.bind(seq_len) if is_prefill else seq_len
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
    q = g.reshape(q, (num_heads * qk_head_dim, _S))
    q = g.reshape(q, (qk_head_dim, num_heads, _S))
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
    kv_expanded = g.reshape(kv_expanded, (num_heads * (qk_nope_dim + v_head_dim), _S))
    kv_expanded = g.reshape(kv_expanded, (qk_nope_dim + v_head_dim, num_heads, _S))
    kv_expanded = g.permute(kv_expanded, (0, 2, 1, 3))  # [qk_nope+v, seq, heads]
    k_nope, v = g.slice(kv_expanded, [qk_nope_dim, v_head_dim], dim=0)

    # ---- RoPE ----
    q_rope = g.rope(q_rope, cos, sin, rope_dim=qk_rope_dim)
    k_rope_raw = g.reshape(k_rope_raw, (qk_rope_dim, _S, 1))
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
    attn_out = g.reshape(attn_out, (num_heads * v_head_dim, _S))
    w_o = g.weight(os.path.join(weights_dir, f"{pfx.replace('.', '_')}_o_proj_weight.weights"),
                   (hidden_size, num_heads * v_head_dim), Precision.FP16)
    out = g.matmul(attn_out, w_o)
    x = g.add(x, out)

    # ---- MLP (SwiGLU) ----
    mlp_pfx = f'model.layers.{layer_idx}.mlp'

    w_ln2 = g.weight(os.path.join(weights_dir, f"model_layers_{layer_idx}_post_attention_layernorm_weight.weights"),
                     (hidden_size,), Precision.FP32)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    w_gate_up = g.weight(os.path.join(weights_dir, f"{mlp_pfx.replace('.', '_')}_gate_up_proj_weight.weights"),
                          (2 * cfg_intermediate_size(cfg), hidden_size), Precision.FP16)
    w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx.replace('.', '_')}_down_proj_weight.weights"),
                      (hidden_size, cfg_intermediate_size(cfg)), Precision.FP16)

    # Single merged matmul → slice into gate/up halves (slice is zero-copy view).
    # SILU/MUL are stride-aware and read the view directly (no materialize).
    # Halves A-pack overhead vs two separate matmuls.
    #
    # Fused SILU: applied in the matmul writeback to the gate half of the
    # output (columns [0, intermediate)). Saves a standalone SILU op dispatch.
    intermediate = cfg_intermediate_size(cfg)
    merged = g.matmul(x_normed2, w_gate_up,
                      activation=Activation.SILU,
                      act_n_begin=0, act_n_len=intermediate)
    gate, up = g.slice(merged, [intermediate, intermediate], dim=0)
    # gate already has silu applied — no standalone SILU op
    mlp_hidden = g.mul(gate, up)
    mlp_out = g.matmul(mlp_hidden, w_down)
    x = g.add(x, mlp_out)

    return x, ck_out, cv_out


def cfg_intermediate_size(cfg: dict) -> int:
    return cfg.get('intermediate_size', cfg['hidden_size'] * 3)


def convert_mla(model_dir: str, output_path: str, num_layers: int | None = None,
                prefill_seq_len: int = 128, n_ctx: int = 4096,
                quant: str = "fp16"):
    """Main entry point: export weights + build graphs → single .mollm file."""
    model_dir = Path(model_dir)
    quant = _canonical_quant(quant)
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="mollm_weights_")
    weights_dir = tmp_dir
    weights_rel = "."

    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)
    config_num_layers = cfg.get('num_hidden_layers', num_layers if num_layers is not None else 32)
    if num_layers is not None and num_layers != config_num_layers:
        print(f"Warning: ignoring deprecated num_layers={num_layers}; "
              f"using config.json num_hidden_layers={config_num_layers}")
    num_layers = config_num_layers

    weights = load_safetensors(str(model_dir / 'model.safetensors'))

    # ---- Step 1: Export shared weights to temp dir ----
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
    metadata = {
        "model_name": f"Youtu-LLM-{num_layers}L",
        "architecture": "mla",
        "num_layers": num_layers,
        "hidden_size": cfg['hidden_size'],
        "num_heads": cfg['num_attention_heads'],
        "num_kv_heads": cfg.get('num_key_value_heads', cfg['num_attention_heads']),
        "head_dim": cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads']),
        "prefill_seq_len": prefill_seq_len,
        "n_ctx": n_ctx,
        "vocab_size": cfg['vocab_size'],
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
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [quant=fp16|w8pc|w4g128|w4mixg128]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    quant = sys.argv[3] if len(sys.argv) > 3 else "fp16"
    convert_mla(model_dir, output_path, quant=quant)
