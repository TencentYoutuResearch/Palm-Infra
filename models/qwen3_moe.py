"""Generic converter for Qwen3-style decoder MoE text models.

This converter is intentionally named by architecture, not by any downstream
fork. It supports:
  - Qwen3 decoder full attention with GQA
  - dense MLP layers before first_k_dense_replace
  - routed MoE layers with per-expert HF weights
  - sigmoid router scoring with optional correction bias
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from qwen35 import _canonical_quant, _quant_spec
from transpile import (
    GraphBuilder,
    Precision,
    _write_weight_file,
    quantize_weight_w8_group,
    save_package,
    write_quantized_weight_file_cpp,
)


ROUTER_SCORE_SOFTMAX = 0
ROUTER_SCORE_SIGMOID = 1


def _num_experts(cfg: dict) -> int:
    if "n_routed_experts" in cfg:
        return int(cfg["n_routed_experts"])
    return int(cfg["num_experts"])


def _norm_suffixes(cfg: dict) -> tuple[str, str]:
    kind = cfg.get("_attn_norm_kind", "q_layernorm")
    if kind == "q_norm":
        return "q_norm", "k_norm"
    return "q_layernorm", "k_layernorm"


def _has_router_bias(cfg: dict) -> bool:
    return bool(cfg.get("_has_router_correction_bias", False))


def _router_score_func(cfg: dict) -> int:
    scoring_func = cfg.get("scoring_func")
    if scoring_func == "sigmoid":
        return ROUTER_SCORE_SIGMOID
    if scoring_func == "softmax":
        return ROUTER_SCORE_SOFTMAX
    return ROUTER_SCORE_SIGMOID if _has_router_bias(cfg) else ROUTER_SCORE_SOFTMAX


def _prepare_config_for_weights(model_dir: Path, cfg: dict) -> dict:
    cfg = dict(cfg)
    weight_names = set(_weight_map(model_dir))
    has_q_norm = "model.layers.0.self_attn.q_norm.weight" in weight_names
    has_q_layernorm = "model.layers.0.self_attn.q_layernorm.weight" in weight_names
    if has_q_norm:
        cfg["_attn_norm_kind"] = "q_norm"
    elif has_q_layernorm:
        cfg["_attn_norm_kind"] = "q_layernorm"
    else:
        raise KeyError("missing Q/K attention norm weights")
    first_moe_layer = int(cfg.get("first_k_dense_replace", 0))
    cfg["_has_router_correction_bias"] = (
        f"model.layers.{first_moe_layer}.mlp.gate.e_score_correction_bias"
        in weight_names
    )
    return cfg


def _as_fp32(dtype_str: str, arr: np.ndarray) -> np.ndarray:
    if dtype_str == "BF16":
        as_u32 = arr.astype(np.uint32) << 16
        return as_u32.view(np.float32)
    if dtype_str == "F16":
        return arr.astype(np.float32)
    if dtype_str == "F32":
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    raise ValueError(f"unsupported safetensors dtype: {dtype_str}")


def _as_fp16(dtype_str: str, arr: np.ndarray) -> np.ndarray:
    if dtype_str == "F16":
        return arr.astype(np.float16) if arr.dtype != np.float16 else arr
    return _as_fp32(dtype_str, arr).astype(np.float16)


def _weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    files = sorted(model_dir.glob("model-*.safetensors"))
    if not files:
        files = sorted(model_dir.glob("model.safetensors-*.safetensors"))
    if not files:
        files = list(model_dir.glob("model.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors file in {model_dir}")

    out: dict[str, str] = {}
    for path in files:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        for name in header:
            if name != "__metadata__":
                out[name] = path.name
    return out


def _selected_safetensors(model_dir: Path, wanted: set[str]):
    weight_map = _weight_map(model_dir)
    missing = sorted(wanted - set(weight_map))
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"missing required tensors: {preview}")

    by_file: dict[str, list[str]] = defaultdict(list)
    for name in sorted(wanted):
        by_file[weight_map[name]].append(name)

    for fname, names in sorted(by_file.items()):
        path = model_dir / fname
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            data_base = 8 + header_len
            for name in names:
                meta = header[name]
                dtype_str = meta["dtype"]
                shape = meta["shape"]
                begin, end = meta["data_offsets"]
                f.seek(data_base + begin)
                raw = f.read(end - begin)
                np_dtype = {
                    "F32": np.float32,
                    "F16": np.float16,
                    "BF16": np.uint16,
                }[dtype_str]
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                yield name, dtype_str, arr


def _write_maybe_quantized(path: str, data: np.ndarray, quant: str,
                           quantizable: bool, raw_name: str,
                           quant_counts: dict[str, int]):
    qspec = (
        _quant_spec(quant, data.shape[1], raw_name)
        if quantizable and data.ndim == 2 else None
    )
    if qspec is not None:
        quant_kind, group_size = qspec
        if write_quantized_weight_file_cpp(
            path, data, quant_kind, group_size, required=(quant_kind == "w4")
        ):
            quant_counts[quant_kind] += 1
            return
        if quant_kind == "w8":
            q, scales, gs, ng = quantize_weight_w8_group(data, group_size)
            _write_weight_file(path, q, scales=scales,
                               group_size=gs, num_groups=ng)
            quant_counts[quant_kind] += 1
            return
        raise ValueError(f"unsupported quant kind for {raw_name}: {quant_kind}")

    _write_weight_file(path, data)


def _non_expert_weight_names(cfg: dict, num_layers: int,
                             available: set[str]) -> set[str]:
    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    q_norm_name, k_norm_name = _norm_suffixes(cfg)
    has_router_bias = _has_router_bias(cfg)
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        attn = f"{pfx}.self_attn"
        names.update({
            f"{pfx}.input_layernorm.weight",
            f"{pfx}.post_attention_layernorm.weight",
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
            f"{attn}.o_proj.weight",
            f"{attn}.{q_norm_name}.weight",
            f"{attn}.{k_norm_name}.weight",
        })
        mlp = f"{pfx}.mlp"
        if i < dense_until:
            names.update({
                f"{mlp}.gate_proj.weight",
                f"{mlp}.up_proj.weight",
                f"{mlp}.down_proj.weight",
            })
        else:
            names.add(f"{mlp}.gate.weight")
            if has_router_bias:
                names.add(f"{mlp}.gate.e_score_correction_bias")
            if int(cfg.get("n_shared_experts", 0) or 0) > 0:
                names.update({
                    f"{mlp}.shared_experts.gate_proj.weight",
                    f"{mlp}.shared_experts.up_proj.weight",
                    f"{mlp}.shared_experts.down_proj.weight",
                })
    return {name for name in names if name in available}


def _expert_weight_names(layer_idx: int, num_experts: int) -> set[str]:
    mlp = f"model.layers.{layer_idx}.mlp"
    names: set[str] = set()
    for e in range(num_experts):
        ep = f"{mlp}.experts.{e}"
        names.update({
            f"{ep}.gate_proj.weight",
            f"{ep}.up_proj.weight",
            f"{ep}.down_proj.weight",
        })
    return names


def _model_name_from_config(cfg: dict, fallback: str) -> str:
    for key in ("model_name", "_name_or_path", "name_or_path",
                "base_model_name_or_path"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip().rstrip("/\\")).name
    return fallback


def _fallback_model_name(model_dir: Path, cfg: dict, num_layers: int) -> str:
    if cfg.get("model_type") == "qwen3_moe" and model_dir.name:
        return model_dir.name
    return f"Qwen3-MoE-{num_layers}L"


def export_weights(model_dir: Path, weights_dir: str, cfg: dict,
                   num_layers: int, quant: str = "fp16"):
    os.makedirs(weights_dir, exist_ok=True)
    quant = _canonical_quant(quant)
    quant_counts = {"w4": 0, "w8": 0}
    hidden_size = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    n_shared = int(cfg.get("n_shared_experts", 0) or 0)
    weight_map = _weight_map(model_dir)

    def save(name: str, data: np.ndarray, quantizable: bool = False,
             raw_name: str = ""):
        _write_maybe_quantized(
            os.path.join(weights_dir, f"{name}.weights"),
            data, quant, quantizable, raw_name or name, quant_counts)

    print("  Exporting non-expert weights...")
    for wname, dtype_str, wdata in _selected_safetensors(
            model_dir, _non_expert_weight_names(cfg, num_layers,
                                                set(weight_map))):
        if wname == "model.embed_tokens.weight":
            save("embed_tokens", _as_fp16(dtype_str, wdata))
            continue
        if wname == "lm_head.weight":
            save("lm_head", _as_fp16(dtype_str, wdata),
                 quantizable=True, raw_name=wname)
            continue
        if wname == "model.norm.weight":
            save("final_norm", _as_fp32(dtype_str, wdata))
            continue
        if "norm" in wname.lower() or "layernorm" in wname.lower():
            save(wname.replace(".", "_"), _as_fp32(dtype_str, wdata))
            continue
        if wname.endswith(".gate.e_score_correction_bias"):
            save(wname.replace(".", "_"), _as_fp32(dtype_str, wdata))
            continue
        if wname.endswith(".mlp.gate.weight"):
            save(wname.replace(".", "_"), _as_fp16(dtype_str, wdata),
                 quantizable=False, raw_name=wname)
            continue
        d = _as_fp16(dtype_str, wdata)
        save(wname.replace(".", "_"), d, quantizable=True, raw_name=wname)

    for layer_idx in range(dense_until, num_layers):
        print(f"  Exporting MoE experts layer {layer_idx}/{num_layers - 1}...")
        gate_up = np.empty((num_experts * 2 * moe_intermediate, hidden_size),
                           dtype=np.float16)
        down = np.empty((num_experts * hidden_size, moe_intermediate),
                        dtype=np.float16)
        for wname, dtype_str, wdata in _selected_safetensors(
                model_dir, _expert_weight_names(layer_idx, num_experts)):
            parts = wname.split(".")
            expert_idx = int(parts[5])
            kind = parts[6]
            d = _as_fp16(dtype_str, wdata)
            if kind == "gate_proj":
                row = expert_idx * 2 * moe_intermediate
                gate_up[row:row + moe_intermediate, :] = d
            elif kind == "up_proj":
                row = expert_idx * 2 * moe_intermediate + moe_intermediate
                gate_up[row:row + moe_intermediate, :] = d
            elif kind == "down_proj":
                row = expert_idx * hidden_size
                down[row:row + hidden_size, :] = d

        mlp_name = f"model_layers_{layer_idx}_mlp"
        save(f"{mlp_name}_experts_gate_up_proj", gate_up,
             quantizable=True,
             raw_name=f"model.layers.{layer_idx}.mlp.experts.gate_up")
        save(f"{mlp_name}_experts_down_proj", down,
             quantizable=True,
             raw_name=f"model.layers.{layer_idx}.mlp.experts.down")

    if n_shared > 0:
        print("  Shared experts are exported as regular shared_experts weights.")

    if quant != "fp16":
        print(f"  Quantized tensors: W4={quant_counts['w4']} W8={quant_counts['w8']}")


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 16384, is_prefill: bool = False) -> GraphBuilder:
    g = GraphBuilder()
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))
    intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    top_k = int(cfg["num_experts_per_tok"])
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    n_shared = int(cfg.get("n_shared_experts", 0) or 0)
    shared_intermediate = n_shared * moe_intermediate
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_interleave = bool(cfg.get("rope_interleave", False))
    rope_dim = head_dim
    vocab_size = int(cfg["vocab_size"])
    router_score_func = _router_score_func(cfg)

    print(f"Qwen3-MoE graph: seq_len={seq_len}, layers={num_layers}, "
          f"dense_until={dense_until}, experts={num_experts}, top_k={top_k}, "
          f"moe_intermediate={moe_intermediate}")

    g.set_model_config(
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        model_type="qwen3_moe",
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
            g, x, i, weights_dir, cos, sin, mask, ck, cv, eps, seq_len,
            num_heads, num_kv_heads, head_dim, hidden_size, intermediate,
            moe_intermediate, num_experts, top_k, dense_until,
            shared_intermediate, n_shared, _has_router_bias(cfg),
            router_score_func,
            bool(cfg.get("norm_topk_prob", True)),
            int(cfg.get("n_group", 1)),
            int(cfg.get("topk_group", 1)),
            float(cfg.get("routed_scaling_factor", 1.0)),
            str(cfg.get("_attn_norm_kind", "q_layernorm")),
            rope_interleave,
            is_prefill=is_prefill)

    w_norm = g.weight(os.path.join(weights_dir, "final_norm.weights"),
                      (hidden_size,), Precision.FP32)
    x = g.rms_norm(x, w_norm, eps=eps)

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def _build_layer(g: GraphBuilder, x: int, layer_idx: int, weights_dir: str,
                 cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
                 eps: float, seq_len: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, hidden_size: int, intermediate: int,
                 moe_intermediate: int, num_experts: int, top_k: int,
                 dense_until: int, shared_intermediate: int, n_shared: int,
                 has_router_bias: bool, router_score_func: int,
                 norm_topk_prob: bool, n_group: int, topk_group: int,
                 routed_scaling_factor: float, attn_norm_kind: str,
                 rope_interleave: bool, is_prefill: bool = False) -> int:
    pfx = f"model_layers_{layer_idx}"

    w_ln = g.weight(os.path.join(weights_dir, f"{pfx}_input_layernorm_weight.weights"),
                    (hidden_size,), Precision.FP32)
    x_normed = g.rms_norm(x, w_ln, eps=eps)

    attn_out = _build_attention(
        g, x_normed, layer_idx, weights_dir, cos, sin, mask, ck_in, cv_in,
        eps, seq_len, num_heads, num_kv_heads, head_dim, hidden_size,
        attn_norm_kind, rope_interleave, is_prefill=is_prefill)
    x = g.add(x, attn_out)

    w_ln2 = g.weight(os.path.join(weights_dir, f"{pfx}_post_attention_layernorm_weight.weights"),
                     (hidden_size,), Precision.FP32)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    mlp_pfx = f"{pfx}_mlp"
    if layer_idx < dense_until:
        w_gate = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_proj_weight.weights"),
                          (intermediate, hidden_size), Precision.FP16)
        w_up = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_up_proj_weight.weights"),
                        (intermediate, hidden_size), Precision.FP16)
        w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_down_proj_weight.weights"),
                          (hidden_size, intermediate), Precision.FP16)
        gate = g.silu(g.matmul(x_normed2, w_gate))
        up = g.matmul(x_normed2, w_up)
        return g.add(x, g.matmul(g.mul(gate, up), w_down))

    w_router = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_weight.weights"),
                        (num_experts, hidden_size), Precision.FP16)
    w_bias = None
    if has_router_bias:
        w_bias = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_e_score_correction_bias.weights"),
                          (num_experts,), Precision.FP32)
    w_experts_gate_up = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_experts_gate_up_proj.weights"),
        (num_experts * 2 * moe_intermediate, hidden_size), Precision.FP16)
    w_experts_down = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_experts_down_proj.weights"),
        (num_experts * hidden_size, moe_intermediate), Precision.FP16)

    has_shared = n_shared > 0
    if has_shared:
        w_shared_gate = g.weight(
            os.path.join(weights_dir, f"{mlp_pfx}_shared_experts_gate_proj_weight.weights"),
            (shared_intermediate, hidden_size), Precision.FP16)
        w_shared_up = g.weight(
            os.path.join(weights_dir, f"{mlp_pfx}_shared_experts_up_proj_weight.weights"),
            (shared_intermediate, hidden_size), Precision.FP16)
        w_shared_down = g.weight(
            os.path.join(weights_dir, f"{mlp_pfx}_shared_experts_down_proj_weight.weights"),
            (hidden_size, shared_intermediate), Precision.FP16)
    else:
        w_shared_gate = g.constant(np.zeros((1, hidden_size), dtype=np.float16))
        w_shared_up = g.constant(np.zeros((1, hidden_size), dtype=np.float16))
        w_shared_down = g.constant(np.zeros((hidden_size, 1), dtype=np.float16))
    w_shared_expert_gate = g.constant(np.zeros((1, hidden_size), dtype=np.float16))

    mlp_out = g.moe(
        x_normed2, w_router, w_experts_gate_up, w_experts_down,
        w_shared_gate, w_shared_up, w_shared_down, w_shared_expert_gate,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=moe_intermediate,
        shared_intermediate_size=shared_intermediate,
        router_bias=w_bias,
        router_score_func=router_score_func,
        norm_topk_prob=norm_topk_prob,
        has_shared_expert=has_shared,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor)
    return g.add(x, mlp_out)


def _build_attention(g: GraphBuilder, x: int, layer_idx: int, weights_dir: str,
                     cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
                     eps: float, seq_len: int, num_heads: int,
                     num_kv_heads: int, head_dim: int, hidden_size: int,
                     attn_norm_kind: str, rope_interleave: bool,
                     is_prefill: bool = False) -> int:
    from transpile import SEQ as SEQ_SYMBOL

    _S = SEQ_SYMBOL.bind(seq_len) if is_prefill else seq_len
    pfx = f"model_layers_{layer_idx}_self_attn"

    w_q = g.weight(os.path.join(weights_dir, f"{pfx}_q_proj_weight.weights"),
                   (num_heads * head_dim, hidden_size), Precision.FP16)
    w_k = g.weight(os.path.join(weights_dir, f"{pfx}_k_proj_weight.weights"),
                   (num_kv_heads * head_dim, hidden_size), Precision.FP16)
    w_v = g.weight(os.path.join(weights_dir, f"{pfx}_v_proj_weight.weights"),
                   (num_kv_heads * head_dim, hidden_size), Precision.FP16)

    query = g.matmul(x, w_q)
    k = g.matmul(x, w_k)
    v = g.matmul(x, w_v)

    q_norm_name, k_norm_name = _norm_suffixes({
        "_attn_norm_kind": attn_norm_kind,
    })

    w_qn = g.weight(os.path.join(weights_dir, f"{pfx}_{q_norm_name}_weight.weights"),
                    (head_dim,), Precision.FP32)
    w_kn = g.weight(os.path.join(weights_dir, f"{pfx}_{k_norm_name}_weight.weights"),
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

    query = g.rope(query, cos, sin, rope_dim=head_dim, interleave=rope_interleave)
    k = g.rope(k, cos, sin, rope_dim=head_dim, interleave=rope_interleave)

    attn, _, _ = g.sdpa(
        query, k, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=head_dim ** -0.5,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, v_head_dim=head_dim)

    attn = g.permute(attn, (0, 2, 1, 3))
    attn = g.contiguous(attn)
    attn = g.reshape(attn, (num_heads * head_dim, _S))

    w_o = g.weight(os.path.join(weights_dir, f"{pfx}_o_proj_weight.weights"),
                   (hidden_size, num_heads * head_dim), Precision.FP16)
    return g.matmul(attn, w_o)


def _moe_expert_storage_metadata(weights_dir: str, cfg: dict,
                                 num_layers: int) -> dict:
    hidden_size = int(cfg["hidden_size"])
    intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    layers = []
    for layer_idx in range(dense_until, num_layers):
        mlp_pfx = f"model_layers_{layer_idx}_mlp"
        layers.append({
            "layer": layer_idx,
            "num_experts": num_experts,
            "gate_up": {
                "weight": os.path.join(weights_dir, f"{mlp_pfx}_experts_gate_up_proj.weights"),
                "rows_per_expert": 2 * intermediate,
                "cols": hidden_size,
                "logical_shape": [num_experts, 2 * intermediate, hidden_size],
            },
            "down": {
                "weight": os.path.join(weights_dir, f"{mlp_pfx}_experts_down_proj.weights"),
                "rows_per_expert": hidden_size,
                "cols": intermediate,
                "logical_shape": [num_experts, hidden_size, intermediate],
            },
        })
    return {
        "version": 1,
        "layout": "aggregate_rows_v1",
        "num_experts": num_experts,
        "layers": layers,
    }


def convert_qwen3_moe(model_dir: str, output_path: str,
                      num_layers: int | None = None,
                      prefill_seq_len: int = 256,
                      n_ctx: int = 16384,
                      quant: str = "fp16"):
    model_dir = Path(model_dir)
    quant = _canonical_quant(quant)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    cfg = _prepare_config_for_weights(model_dir, cfg)

    config_num_layers = int(cfg["num_hidden_layers"])
    if num_layers is None:
        num_layers = config_num_layers
    if num_layers <= 0 or num_layers > config_num_layers:
        raise ValueError(f"num_layers must be in [1, {config_num_layers}], got {num_layers}")
    if num_layers != config_num_layers:
        print(f"Debug: truncating Qwen3-MoE layers to {num_layers}/{config_num_layers}")
    cfg["num_hidden_layers"] = num_layers

    tmp_dir = tempfile.mkdtemp(prefix="mollm_qwen3_moe_weights_")
    weights_dir = tmp_dir
    weights_rel = "."
    try:
        print("Exporting selected Qwen3-MoE weights...")
        export_weights(model_dir, weights_dir, cfg, num_layers, quant=quant)

        print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
        g_prefill = build_graph(weights_rel, cfg, seq_len=prefill_seq_len,
                                n_ctx=n_ctx, is_prefill=True)

        print("\nBuilding decode graph (seq_len=1)...")
        g_decode = build_graph(weights_rel, cfg, seq_len=1, n_ctx=n_ctx)

        fallback_model_name = _fallback_model_name(model_dir, cfg, num_layers)
        metadata = {
            "model_name": _model_name_from_config(cfg, fallback_model_name),
            "architecture": "qwen3-moe",
            "num_layers": num_layers,
            "hidden_size": cfg["hidden_size"],
            "num_heads": cfg["num_attention_heads"],
            "num_kv_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"],
            "prefill_seq_len": prefill_seq_len,
            "n_ctx": n_ctx,
            "vocab_size": cfg["vocab_size"],
            "num_experts": _num_experts(cfg),
            "num_experts_per_tok": cfg["num_experts_per_tok"],
            "moe_intermediate_size": cfg["moe_intermediate_size"],
            "first_k_dense_replace": cfg.get("first_k_dense_replace", 0),
            "n_shared_experts": cfg.get("n_shared_experts", 0),
            "moe_expert_storage": _moe_expert_storage_metadata(weights_rel, cfg, num_layers),
            "quantization": quant,
        }

        print(f"\nPacking {output_path}...")
        save_package(output_path, g_prefill, g_decode, weights_dir, metadata,
                     tokenizer_path=str(model_dir / "tokenizer.json"),
                     jinja_path=str(model_dir / "chat_template.jinja"))
        print(f"\nDone! Output: {output_path}")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Qwen3-MoE-compatible models to .mollm")
    parser.add_argument("model_dir")
    parser.add_argument("output")
    parser.add_argument("quant", nargs="?", default="fp16")
    parser.add_argument("--layers", type=int, default=None,
                        help="debug-only layer truncation")
    parser.add_argument("--prefill-seq-len", type=int, default=256,
                        help="prefill graph chunk length")
    args = parser.parse_args()
    convert_qwen3_moe(args.model_dir, args.output,
                      num_layers=args.layers,
                      prefill_seq_len=args.prefill_seq_len,
                      quant=args.quant)
