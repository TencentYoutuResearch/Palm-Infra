"""mollm model builder for standard Qwen3 decoder-only text models.

This path covers models such as Qwen3-4B-Instruct-2507:
  - regular full attention in every layer
  - GQA via num_attention_heads / num_key_value_heads
  - QK RMSNorm
  - SwiGLU MLP

It intentionally stays separate from qwen35.py, whose graph is for the
Qwen3.5 hybrid linear/full-attention architecture.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from qwen35 import (
    _canonical_quant,
    _quant_spec,
    load_safetensors,
)
from transpile import (
    GraphBuilder,
    Precision,
    SEQ,
    _write_weight_file,
    quantize_weight_w8_group,
    save_package,
    write_quantized_weight_file_cpp,
)
from model_metadata import infer_hf_model_name


def _safetensor_files(model_dir: Path) -> list[Path]:
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors-*.safetensors",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(model_dir.glob(pattern)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def export_weights(weights: dict[str, np.ndarray], weights_dir: str,
                   quant: str = "fp16"):
    os.makedirs(weights_dir, exist_ok=True)
    quant = _canonical_quant(quant)
    quant_counts = {"w4": 0, "w8": 0}
    lm_head_source = None

    def save(name: str, data: np.ndarray, quantizable: bool = False,
             raw_name: str = ""):
        wpath = os.path.join(weights_dir, f"{name}.weights")
        quant_spec = (
            _quant_spec(quant, data.shape[1], raw_name)
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
                _write_weight_file(wpath, q, scales=scales,
                                   group_size=gs, num_groups=ng)
                quant_counts[quant_kind] += 1
                return
            raise ValueError(f"unsupported quant kind: {quant_kind}")

        _write_weight_file(wpath, data)

    for wname, wdata in weights.items():
        if wname == "lm_head.weight":
            lm_head_source = wdata
            continue

        if wname == "model.embed_tokens.weight":
            if lm_head_source is None:
                lm_head_source = wdata
            d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
            save("embed_tokens", d)
            continue

        if wname == "model.norm.weight":
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            save("final_norm", d)
            continue

        if "model.layers." not in wname:
            continue

        if "norm" in wname.lower() or "layernorm" in wname.lower():
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            save(wname.replace(".", "_"), d)
            continue

        # Skip mlp.gate_proj / mlp.up_proj — merged below into gate_up_proj so the
        # graph runs one large matmul + a fused SWIGLU op.
        if ".mlp.gate_proj." in wname or ".mlp.up_proj." in wname:
            continue
        # Skip self_attn q/k/v_proj — merged below into qkv_proj (one matmul + slice).
        if (".self_attn.q_proj." in wname or ".self_attn.k_proj." in wname
                or ".self_attn.v_proj." in wname):
            continue

        d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
        save(wname.replace(".", "_"), d, quantizable=True, raw_name=wname)

    # Merge self_attn q/k/v_proj per layer into one [q+k+v, hidden] weight
    # (concat along dim 0 = output/N). One matmul + slice at inference.
    import re as _re_qkv
    qkv_pat = _re_qkv.compile(r'^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$')
    qkv_ids = sorted({int(m.group(1)) for wname in weights
                      for m in [qkv_pat.match(wname)] if m})
    for layer_idx in qkv_ids:
        qn = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        kn = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        vn = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        if kn not in weights or vn not in weights:
            continue
        merged = np.concatenate([weights[qn], weights[kn], weights[vn]], axis=0)
        if merged.dtype != np.float16:
            merged = merged.astype(np.float16)
        save(f"model_layers_{layer_idx}_self_attn_qkv_proj_weight", merged,
             quantizable=True,
             raw_name=f"model.layers.{layer_idx}.self_attn.qkv_proj.weight")

    # Merge mlp.gate_proj + mlp.up_proj per layer into one [2*intermediate, hidden]
    # weight (concat along dim 0 = output/N). One matmul + SWIGLU at inference.
    import re
    layer_pat = re.compile(r'^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$')
    layer_ids = sorted({int(m.group(1)) for wname in weights
                        for m in [layer_pat.match(wname)] if m})
    for layer_idx in layer_ids:
        gate_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        up_name   = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        if up_name not in weights:
            continue
        merged = np.concatenate([weights[gate_name], weights[up_name]], axis=0)
        if merged.dtype != np.float16:
            merged = merged.astype(np.float16)
        save(f"model_layers_{layer_idx}_mlp_gate_up_proj_weight", merged,
             quantizable=True,
             raw_name=f"model.layers.{layer_idx}.mlp.gate_up_proj.weight")

    if lm_head_source is None:
        raise KeyError("No lm_head.weight or model.embed_tokens.weight found")
    d = lm_head_source.astype(np.float16) if lm_head_source.dtype != np.float16 else lm_head_source
    save("lm_head", d, quantizable=True, raw_name="lm_head.weight")

    if quant != "fp16":
        print(f"  Quantized tensors: W4={quant_counts['w4']} W8={quant_counts['w8']}")


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 16384, is_prefill: bool = False) -> GraphBuilder:
    g = GraphBuilder()

    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    eps = cfg.get("rms_norm_eps", 1e-6)
    rope_theta = cfg.get("rope_theta", 1000000.0)
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", hidden_size // num_heads)
    intermediate = cfg["intermediate_size"]
    vocab_size = cfg["vocab_size"]
    rope_dim = head_dim

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads must be divisible by num_key_value_heads: "
            f"{num_heads} vs {num_kv_heads}"
        )
    if num_heads * head_dim != hidden_size and num_heads * head_dim <= 0:
        raise ValueError("invalid Qwen3 attention head configuration")

    print(f"Qwen3 graph: seq_len={seq_len}, layers={num_layers}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"hidden={hidden_size}, intermediate={intermediate}")

    g.set_model_config(
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        model_type="qwen3",
    )

    embed_shape = (vocab_size, hidden_size)
    g.weight(os.path.join(weights_dir, "embed_tokens.weights"),
             embed_shape, Precision.FP16)
    g.weight(os.path.join(weights_dir, "lm_head.weights"),
             embed_shape, Precision.FP16)

    if is_prefill:
        from transpile import DimExpr
        CONST = DimExpr.const()
        SEQ_DIM = DimExpr.seq()
        hidden_dyn = (CONST, SEQ_DIM, CONST, CONST)
        mask_dyn = (CONST, SEQ_DIM, CONST, CONST)
        cos_dyn = (CONST, SEQ_DIM, CONST, CONST)
        sin_dyn = (CONST, SEQ_DIM, CONST, CONST)
    else:
        hidden_dyn = mask_dyn = cos_dyn = sin_dyn = None

    hidden = g.input("hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    mask = g.input("mask", (1, seq_len), dynamic=mask_dyn)
    cos = g.input("cos", (rope_dim // 2, seq_len), dynamic=cos_dyn)
    sin = g.input("sin", (rope_dim // 2, seq_len), dynamic=sin_dyn)

    caches = []
    for i in range(num_layers):
        ck = g.input(f"cache_k{i}", (head_dim, n_ctx, num_kv_heads),
                     prec=Precision.FP16)
        cv = g.input(f"cache_v{i}", (head_dim, n_ctx, num_kv_heads),
                     prec=Precision.FP16)
        caches.append((ck, cv))

    x = hidden
    for i in range(num_layers):
        ck, cv = caches[i]
        x = _build_layer(
            g, x, i, weights_dir, cos, sin, mask, ck, cv,
            eps, seq_len, num_heads, num_kv_heads, head_dim,
            intermediate, hidden_size, is_prefill=is_prefill,
        )

    w_norm = g.weight(os.path.join(weights_dir, "final_norm.weights"),
                      (hidden_size,), Precision.FP32)
    x = g.rms_norm(x, w_norm, eps=eps)

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def _build_layer(g: GraphBuilder, x: int, layer_idx: int, weights_dir: str,
                 cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
                 eps: float, seq_len: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate: int, hidden_size: int,
                 is_prefill: bool = False) -> int:
    pfx = f"model_layers_{layer_idx}"

    w_ln = g.weight(os.path.join(weights_dir, f"{pfx}_input_layernorm_weight.weights"),
                    (hidden_size,), Precision.FP32)
    x_normed = g.rms_norm(x, w_ln, eps=eps)

    attn_out = _build_attention(
        g, x_normed, layer_idx, weights_dir, cos, sin, mask, ck_in, cv_in,
        eps, seq_len, num_heads, num_kv_heads, head_dim, hidden_size,
        is_prefill=is_prefill,
    )
    x = g.add(x, attn_out)

    w_ln2 = g.weight(os.path.join(weights_dir, f"{pfx}_post_attention_layernorm_weight.weights"),
                     (hidden_size,), Precision.FP32)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    mlp_pfx = f"{pfx}_mlp"
    # Fused gate/up: one [2*intermediate, hidden] matmul (activation=NONE so the
    # Metal tensor GEMM fast path is used) + a fused SWIGLU op (silu(gate)*up).
    w_gate_up = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_up_proj_weight.weights"),
                         (2 * intermediate, hidden_size), Precision.FP16)
    w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_down_proj_weight.weights"),
                      (hidden_size, intermediate), Precision.FP16)

    merged = g.matmul(x_normed2, w_gate_up)
    mlp_hidden = g.swiglu(merged)
    mlp_out = g.matmul(mlp_hidden, w_down)
    return g.add(x, mlp_out)


def _build_attention(g: GraphBuilder, x: int, layer_idx: int, weights_dir: str,
                     cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
                     eps: float, seq_len: int, num_heads: int,
                     num_kv_heads: int, head_dim: int, hidden_size: int,
                     is_prefill: bool = False) -> int:
    from transpile import SEQ as SEQ_SYMBOL

    _S = SEQ_SYMBOL.bind(seq_len) if is_prefill else seq_len
    pfx = f"model_layers_{layer_idx}_self_attn"

    # Fused QKV: one [q_dim+k_dim+v_dim, hidden] matmul (activation=NONE → Metal
    # tensor GEMM), then slice into q/k/v. The small k/v projections (N=1024)
    # have poor tensor-engine occupancy alone; sharing one large matmul helps.
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    w_qkv = g.weight(os.path.join(weights_dir, f"{pfx}_qkv_proj_weight.weights"),
                     (q_dim + 2 * kv_dim, hidden_size), Precision.FP16)
    qkv = g.matmul(x, w_qkv)
    query, k, v = g.slice(qkv, [q_dim, kv_dim, kv_dim], dim=0)

    w_qn = g.weight(os.path.join(weights_dir, f"{pfx}_q_norm_weight.weights"),
                    (head_dim,), Precision.FP32)
    w_kn = g.weight(os.path.join(weights_dir, f"{pfx}_k_norm_weight.weights"),
                    (head_dim,), Precision.FP32)

    query = g.reshape(query, (head_dim, num_heads, _S))
    query = g.permute(query, (0, 2, 1, 3))
    query = g.reshape(query, (head_dim, num_heads * _S))
    query = g.rms_norm(query, w_qn, eps=eps)
    query = g.reshape(query, (head_dim, _S, num_heads))
    query = g.contiguous(query)

    k = g.reshape(k, (head_dim, num_kv_heads, _S))
    k = g.permute(k, (0, 2, 1, 3))
    k = g.reshape(k, (head_dim, num_kv_heads * _S))
    k = g.rms_norm(k, w_kn, eps=eps)
    k = g.reshape(k, (head_dim, _S, num_kv_heads))
    k = g.contiguous(k)

    v = g.reshape(v, (head_dim, num_kv_heads, _S))
    v = g.permute(v, (0, 2, 1, 3))
    v = g.contiguous(v)

    query = g.rope(query, cos, sin, rope_dim=head_dim, interleave=False)
    k = g.rope(k, cos, sin, rope_dim=head_dim, interleave=False)

    attn, _, _ = g.sdpa(
        query, k, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=head_dim ** -0.5,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, v_head_dim=head_dim,
    )

    attn = g.permute(attn, (0, 2, 1, 3))
    attn = g.contiguous(attn)
    attn = g.reshape(attn, (num_heads * head_dim, _S))

    w_o = g.weight(os.path.join(weights_dir, f"{pfx}_o_proj_weight.weights"),
                   (hidden_size, num_heads * head_dim), Precision.FP16)
    return g.matmul(attn, w_o)


def convert_qwen3(model_dir: str, output_path: str, num_layers: int | None = None,
                  prefill_seq_len: int = 256, n_ctx: int = 16384,
                  quant: str = "fp16"):
    model_dir = Path(model_dir)
    quant = _canonical_quant(quant)
    tmp_dir = tempfile.mkdtemp(prefix="mollm_qwen3_weights_")
    weights_dir = tmp_dir
    weights_rel = "."

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    config_num_layers = cfg["num_hidden_layers"]
    if num_layers is not None and num_layers != config_num_layers:
        print(f"Warning: ignoring deprecated num_layers={num_layers}; "
              f"using config.json num_hidden_layers={config_num_layers}")
    num_layers = config_num_layers

    st_files = _safetensor_files(model_dir)
    if not st_files:
        raise FileNotFoundError(f"No safetensors file in {model_dir}")

    weights = {}
    for st_path in st_files:
        weights.update(load_safetensors(str(st_path)))

    try:
        print("Exporting weights...")
        export_weights(weights, str(weights_dir), quant=quant)

        print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
        g_prefill = build_graph(weights_rel, cfg, seq_len=prefill_seq_len,
                                n_ctx=n_ctx, is_prefill=True)

        print(f"\nBuilding decode graph (seq_len=1)...")
        g_decode = build_graph(weights_rel, cfg, seq_len=1, n_ctx=n_ctx)

        print(f"\nPacking {output_path}...")
        fallback_model_name = f"Qwen3-{num_layers}L"
        metadata = {
            "model_name": infer_hf_model_name(model_dir, cfg, fallback_model_name),
            "architecture": "qwen3",
            "num_layers": num_layers,
            "hidden_size": cfg["hidden_size"],
            "num_heads": cfg["num_attention_heads"],
            "num_kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            "head_dim": cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            "prefill_seq_len": prefill_seq_len,
            "n_ctx": n_ctx,
            "vocab_size": cfg["vocab_size"],
            "quantization": quant,
        }
        save_package(output_path, g_prefill, g_decode, weights_dir, metadata,
                     tokenizer_path=str(model_dir / "tokenizer.json"),
                     jinja_path=str(model_dir / "chat_template.jinja"))
    finally:
        import shutil
        shutil.rmtree(tmp_dir)

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [quant=fp16|w8pc|w4g128|w4mixg128]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    quant = sys.argv[3] if len(sys.argv) > 3 else "fp16"
    convert_qwen3(model_dir, output_path, quant=quant)
