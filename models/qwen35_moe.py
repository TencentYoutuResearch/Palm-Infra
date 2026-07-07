"""
Text-only converter for Qwen3.5-MoE.

This first implementation is correctness-oriented and exports FP16/FP32
weights only. Expert tensors are loaded selectively from sharded safetensors so
1-2 layer packages can be built without materializing the full 35B-A3B model.
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

from transpile import (
    GraphBuilder, Precision, _write_weight_file,
    quantize_weight_w8_group, save_package,
    write_quantized_weight_file_cpp,
)
from qwen35 import _build_full_attn_layer, _build_linear_attn_layer


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


def _quant_spec(quant: str, k: int) -> tuple[str, int] | None:
    if quant == "none":
        return None
    if quant == "w8pc":
        return ("w8", k)
    if quant.startswith("w8g"):
        return ("w8", int(quant[3:]))
    if quant.startswith("w4g"):
        return ("w4", int(quant[3:]))
    if quant.startswith("w4mixg"):
        return ("w4", int(quant[6:]))
    raise ValueError(f"unsupported quant mode: {quant}")


def _selected_safetensors(model_dir: Path, wanted: set[str]):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        files = sorted(model_dir.glob("model.safetensors-*.safetensors"))
        if not files:
            files = list(model_dir.glob("model.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors file in {model_dir}")
        weight_map = {}
        for path in files:
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len).decode("utf-8"))
            for name in header:
                if name != "__metadata__":
                    weight_map[name] = path.name

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


def _required_weight_names(tc: dict, num_layers: int) -> set[str]:
    names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    layer_types = tc["layer_types"]
    for i in range(num_layers):
        pfx = f"model.language_model.layers.{i}"
        names.add(f"{pfx}.input_layernorm.weight")
        names.add(f"{pfx}.post_attention_layernorm.weight")

        if layer_types[i] == "linear_attention":
            lp = f"{pfx}.linear_attn"
            names.update({
                f"{lp}.A_log",
                f"{lp}.conv1d.weight",
                f"{lp}.dt_bias",
                f"{lp}.in_proj_a.weight",
                f"{lp}.in_proj_b.weight",
                f"{lp}.in_proj_qkv.weight",
                f"{lp}.in_proj_z.weight",
                f"{lp}.norm.weight",
                f"{lp}.out_proj.weight",
            })
        else:
            ap = f"{pfx}.self_attn"
            names.update({
                f"{ap}.q_proj.weight",
                f"{ap}.k_proj.weight",
                f"{ap}.v_proj.weight",
                f"{ap}.o_proj.weight",
                f"{ap}.q_norm.weight",
                f"{ap}.k_norm.weight",
            })

        mp = f"{pfx}.mlp"
        names.update({
            f"{mp}.gate.weight",
            f"{mp}.experts.gate_up_proj",
            f"{mp}.experts.down_proj",
            f"{mp}.shared_expert.gate_proj.weight",
            f"{mp}.shared_expert.up_proj.weight",
            f"{mp}.shared_expert.down_proj.weight",
            f"{mp}.shared_expert_gate.weight",
        })
    return names


def export_weights(model_dir: Path, weights_dir: str, cfg: dict, num_layers: int,
                   quant: str = "none"):
    os.makedirs(weights_dir, exist_ok=True)
    tc = cfg["text_config"]
    wanted = _required_weight_names(tc, num_layers)
    quant_counts = {"w4": 0, "w8": 0}

    def save(name: str, data: np.ndarray,
             quantizable: bool = False,
             raw_name: str = ""):
        wpath = os.path.join(weights_dir, f"{name}.weights")
        quant_spec = (
            _quant_spec(quant, data.shape[1])
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
                quant_counts[quant_kind] += 1
                return
            raise ValueError(f"unsupported quant kind for {raw_name or name}: {quant_kind}")

        _write_weight_file(wpath, data)

    for wname, dtype_str, wdata in _selected_safetensors(model_dir, wanted):
        if wname == "lm_head.weight":
            save("lm_head", _as_fp16(dtype_str, wdata),
                 quantizable=True, raw_name=wname)
            continue

        if wname == "model.language_model.embed_tokens.weight":
            save("embed_tokens", _as_fp16(dtype_str, wdata))
            continue

        if wname == "model.language_model.norm.weight":
            d = _as_fp32(dtype_str, wdata)
            save("final_norm", d + 1.0)
            continue

        if "norm" in wname.lower() or "layernorm" in wname.lower():
            d = _as_fp32(dtype_str, wdata)
            if "linear_attn" not in wname:
                d = d + 1.0
            save(wname.replace(".", "_"), d)
            continue

        if "A_log" in wname or "dt_bias" in wname:
            save(wname.replace(".", "_"), _as_fp32(dtype_str, wdata))
            continue

        if "conv1d" in wname:
            d = _as_fp32(dtype_str, wdata).reshape(wdata.shape[0], wdata.shape[2])
            save(wname.replace(".", "_"), d)
            continue

        d = _as_fp16(dtype_str, wdata)
        quantizable = True

        if ".mlp.gate.weight" in wname:
            quantizable = False
        elif ".mlp.shared_expert_gate.weight" in wname:
            quantizable = False
        elif ".mlp.experts.gate_up_proj" in wname:
            d = d.reshape(d.shape[0] * d.shape[1], d.shape[2])
        elif ".mlp.experts.down_proj" in wname:
            d = d.reshape(d.shape[0] * d.shape[1], d.shape[2])

        save(wname.replace(".", "_"), d,
             quantizable=quantizable, raw_name=wname)

    if quant != "none":
        print(f"  Quantized tensors: W4={quant_counts['w4']} W8={quant_counts['w8']}")


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096, is_prefill: bool = False) -> GraphBuilder:
    g = GraphBuilder()
    tc = cfg["text_config"]

    hidden_size = tc["hidden_size"]
    num_layers = tc["num_hidden_layers"]
    layer_types = tc["layer_types"]
    eps = tc.get("rms_norm_eps", 1e-6)
    rope_theta = tc["rope_parameters"]["rope_theta"]
    rope_dim = int(tc["head_dim"] * tc["rope_parameters"]["partial_rotary_factor"])

    num_heads = tc["num_attention_heads"]
    num_kv_heads = tc["num_key_value_heads"]
    head_dim = tc["head_dim"]
    linear_num_heads = tc["linear_num_key_heads"]
    linear_k_dim = tc["linear_key_head_dim"]
    linear_v_dim = tc["linear_value_head_dim"]
    linear_num_v_heads = tc.get("linear_num_value_heads", linear_num_heads)
    conv_kernel = tc["linear_conv_kernel_dim"]

    moe_intermediate = tc["moe_intermediate_size"]
    shared_intermediate = tc["shared_expert_intermediate_size"]
    num_experts = tc["num_experts"]
    top_k = tc["num_experts_per_tok"]

    print(f"Qwen3.5-MoE graph: seq_len={seq_len}, layers={num_layers}, "
          f"experts={num_experts}, top_k={top_k}, moe_intermediate={moe_intermediate}")

    g.set_model_config(
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=tc["vocab_size"],
        model_type="qwen3_5_moe",
    )

    embed_path = os.path.join(weights_dir, "embed_tokens.weights")
    embed_shape = (tc["vocab_size"], hidden_size)
    g.weight(embed_path, embed_shape, Precision.FP16)

    lm_head_path = os.path.join(weights_dir, "lm_head.weights")
    g.weight(lm_head_path, embed_shape, Precision.FP16)

    if is_prefill:
        from transpile import DimExpr
        SEQ_DIM = DimExpr.seq()
        CONST = DimExpr.const()
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

    cache_inputs = []
    for i in range(num_layers):
        lt = layer_types[i]
        if lt == "full_attention":
            ck = g.input(f"cache_k{i}", (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cv = g.input(f"cache_v{i}", (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cache_inputs.append(("kv", ck, cv))
        else:
            gs = g.input(f"gdn_state{i}",
                         (linear_v_dim, linear_k_dim, linear_num_v_heads),
                         prec=Precision.FP32)
            qkv_total = linear_num_heads * linear_k_dim * 2 + linear_num_v_heads * linear_v_dim
            gc = g.input(f"gdn_conv{i}", (qkv_total, conv_kernel - 1), prec=Precision.FP16)
            cache_inputs.append(("gdn", gs, gc))

    x = hidden
    for i in range(num_layers):
        ck_in, cv_in = (cache_inputs[i][1], cache_inputs[i][2]) if cache_inputs[i][0] == "kv" else (None, None)
        gs_in, gc_in = (cache_inputs[i][1], cache_inputs[i][2]) if cache_inputs[i][0] == "gdn" else (None, None)
        x = _build_layer(g, x, i, weights_dir, tc,
                         cos, sin, mask, ck_in, cv_in, gs_in, gc_in,
                         eps, seq_len, rope_dim,
                         num_heads, num_kv_heads, head_dim,
                         linear_num_heads, linear_k_dim, linear_v_dim,
                         linear_num_v_heads, conv_kernel,
                         hidden_size, layer_types[i],
                         moe_intermediate, shared_intermediate,
                         num_experts, top_k,
                         is_prefill=is_prefill)

    w_norm = g.weight(os.path.join(weights_dir, "final_norm.weights"),
                      (hidden_size,), Precision.FP32)
    x = g.rms_norm(x, w_norm, eps=eps)

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def _build_layer(g, x, layer_idx, weights_dir, tc,
                 cos, sin, mask, ck_in, cv_in, gs_in, gc_in,
                 eps, seq_len, rope_dim,
                 num_heads, num_kv_heads, head_dim,
                 linear_num_heads, linear_k_dim, linear_v_dim,
                 linear_num_v_heads, conv_kernel,
                 hidden_size, layer_type,
                 moe_intermediate, shared_intermediate,
                 num_experts, top_k,
                 is_prefill=False):
    pfx_lm = f"model_language_model_layers_{layer_idx}"

    w_ln = g.weight(os.path.join(weights_dir, f"{pfx_lm}_input_layernorm_weight.weights"),
                    (hidden_size,), Precision.FP32)
    x_normed = g.rms_norm(x, w_ln, eps=eps)

    if layer_type == "linear_attention":
        attn_out = _build_linear_attn_layer(g, x_normed, layer_idx, weights_dir,
                                             gs_in, gc_in, eps, seq_len,
                                             linear_num_heads, linear_k_dim,
                                             linear_v_dim, linear_num_v_heads,
                                             conv_kernel, hidden_size)
    else:
        attn_out = _build_full_attn_layer(g, x_normed, layer_idx, weights_dir,
                                           cos, sin, mask, ck_in, cv_in,
                                           eps, seq_len, rope_dim,
                                           num_heads, num_kv_heads, head_dim,
                                           hidden_size, is_prefill=is_prefill)

    x = g.add(x, attn_out)

    w_ln2 = g.weight(os.path.join(weights_dir, f"{pfx_lm}_post_attention_layernorm_weight.weights"),
                     (hidden_size,), Precision.FP32)
    x_normed2 = g.rms_norm(x, w_ln2, eps=eps)

    mlp_pfx = f"{pfx_lm}_mlp"
    w_router = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_weight.weights"),
                        (num_experts, hidden_size), Precision.FP16)
    w_experts_gate_up = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_experts_gate_up_proj.weights"),
        (num_experts * 2 * moe_intermediate, hidden_size), Precision.FP16)
    w_experts_down = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_experts_down_proj.weights"),
        (num_experts * hidden_size, moe_intermediate), Precision.FP16)
    w_shared_gate = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_shared_expert_gate_proj_weight.weights"),
        (shared_intermediate, hidden_size), Precision.FP16)
    w_shared_up = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_shared_expert_up_proj_weight.weights"),
        (shared_intermediate, hidden_size), Precision.FP16)
    w_shared_down = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_shared_expert_down_proj_weight.weights"),
        (hidden_size, shared_intermediate), Precision.FP16)
    w_shared_expert_gate = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_shared_expert_gate_weight.weights"),
        (1, hidden_size), Precision.FP16)

    mlp_out = g.moe(x_normed2, w_router,
                    w_experts_gate_up, w_experts_down,
                    w_shared_gate, w_shared_up, w_shared_down,
                    w_shared_expert_gate,
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    intermediate_size=moe_intermediate,
                    shared_intermediate_size=shared_intermediate)
    return g.add(x, mlp_out)


def convert_qwen35_moe(model_dir: str, output_path: str, num_layers: int = 1,
                       prefill_seq_len: int = 256, n_ctx: int = 4096,
                       quant: str = "none"):
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    cfg = dict(cfg)
    tc = dict(cfg["text_config"])
    if num_layers <= 0 or num_layers > tc["num_hidden_layers"]:
        raise ValueError(f"num_layers must be in [1, {tc['num_hidden_layers']}], got {num_layers}")
    tc["num_hidden_layers"] = num_layers
    tc["layer_types"] = list(tc["layer_types"][:num_layers])
    cfg["text_config"] = tc

    tmp_dir = tempfile.mkdtemp(prefix="mollm_qwen35_moe_weights_")
    weights_dir = tmp_dir
    weights_rel = "."
    try:
        print("Exporting selected Qwen3.5-MoE weights...")
        export_weights(model_dir, weights_dir, cfg, num_layers, quant=quant)

        print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
        g_prefill = build_graph(weights_rel, cfg, seq_len=prefill_seq_len,
                                n_ctx=n_ctx, is_prefill=True)

        print("\nBuilding decode graph (seq_len=1)...")
        g_decode = build_graph(weights_rel, cfg, seq_len=1, n_ctx=n_ctx)

        metadata = {
            "model_name": f"Qwen3.5-MoE-{num_layers}L",
            "architecture": "qwen3.5-moe",
            "num_layers": num_layers,
            "hidden_size": tc["hidden_size"],
            "num_heads": tc["num_attention_heads"],
            "num_kv_heads": tc["num_key_value_heads"],
            "head_dim": tc["head_dim"],
            "prefill_seq_len": prefill_seq_len,
            "n_ctx": n_ctx,
            "vocab_size": tc["vocab_size"],
            "layer_types": tc["layer_types"],
            "num_experts": tc["num_experts"],
            "num_experts_per_tok": tc["num_experts_per_tok"],
            "moe_intermediate_size": tc["moe_intermediate_size"],
            "shared_expert_intermediate_size": tc["shared_expert_intermediate_size"],
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
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [num_layers] [prefill_seq_len] [quant]")
        raise SystemExit(1)
    nl = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ps = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    qt = sys.argv[5] if len(sys.argv) > 5 else "none"
    convert_qwen35_moe(sys.argv[1], sys.argv[2], nl, ps, quant=qt)
