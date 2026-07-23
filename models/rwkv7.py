"""RWKV v7 (.pth) converter.

This intentionally targets the checkpoint format used by the official RWKV
releases rather than requiring a Hugging Face model directory.  The package is
CPU-only for now: WKV state is recurrent and is not an attention KV cache.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from transpile import (
    Activation, DimExpr, GraphBuilder, Precision, _write_weight_file,
    quantize_weight_w8_group, save_package, write_quantized_weight_file_cpp,
)


def _canonical_quant(quant: str) -> str:
    q = quant.lower()
    if q in ("none", "fp16", "f16"):
        return "fp16"
    if q in ("w8", "w8pc"):
        return "w8pc"
    if q in ("w4", "w4mix", "w4mixg128"):
        return "w4mixg128"
    raise ValueError(f"unsupported RWKV quant mode: {quant}")


def _is_small_lora(name: str) -> bool:
    return name.endswith(("_w1", "_w2", "_a1", "_a2", "_v1", "_v2", "_g1", "_g2"))


def _quant_spec(quant: str, name: str, k: int) -> tuple[str, int] | None:
    quant = _canonical_quant(quant)
    if quant == "fp16" or name == "embed_tokens":
        return None
    if quant == "w8pc":
        return ("w8", k)
    # Mixed W4: retain the vocabulary projection and low-rank attention
    # adapters at W8; large attention/FFN matrices use symmetric W4G128.
    if name == "lm_head" or _is_small_lora(name):
        return ("w8", 128)
    return ("w4", 128)


def _matrix_precision(quant: str, name: str, k: int) -> Precision:
    spec = _quant_spec(quant, name, k)
    if spec is None:
        return Precision.FP16
    return Precision.INT8 if spec[0] == "w8" else Precision.INT4


def _torch_load(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("RWKV conversion needs PyTorch to read .pth checkpoints") from exc
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("RWKV checkpoint must be a state_dict")
    out = {k: v.detach().float().cpu().numpy() for k, v in raw.items()
           if hasattr(v, "detach")}
    # Official v7 checkpoints commonly omit layer-0 v* because v_first=value
    # there. Keep the package layout uniform; the recurrence ignores them.
    for k, value in list(out.items()):
        if ".att.a" in k:
            vk = k.replace(".att.a", ".att.v", 1)
            out.setdefault(vk, value.copy())
    return out


def _info(w: dict[str, np.ndarray]) -> tuple[int, int, int, int, int]:
    if "blocks.0.att.r_k" not in w:
        raise ValueError("only RWKV v7 checkpoints are supported (missing blocks.0.att.r_k)")
    layers = max(int(k.split(".")[1]) + 1 for k in w if k.startswith("blocks."))
    vocab, hidden = w["emb.weight"].shape
    heads, head_size = w["blocks.0.att.r_k"].shape
    if heads * head_size != hidden:
        raise ValueError(f"invalid RWKV head configuration: {heads} * {head_size} != {hidden}")
    return layers, vocab, hidden, heads, head_size


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps=1e-5) -> np.ndarray:
    y = x.astype(np.float32)
    return ((y - y.mean(-1, keepdims=True)) /
            np.sqrt(y.var(-1, keepdims=True) + eps) * weight + bias)


def export_weights(w: dict[str, np.ndarray], out: str, quant: str = "fp16") -> None:
    os.makedirs(out, exist_ok=True)
    quant = _canonical_quant(quant)
    layers, _, _, _, _ = _info(w)
    def save_matrix(name: str, arr: np.ndarray) -> None:
        path = os.path.join(out, f"{name}.weights")
        data = arr.astype(np.float16)
        spec = _quant_spec(quant, name, data.shape[1])
        if spec is None:
            _write_weight_file(path, data)
            return
        kind, group = spec
        if write_quantized_weight_file_cpp(path, data, kind, group,
                                           required=(kind == "w4")):
            return
        q, scales, gs, ng = quantize_weight_w8_group(data, group)
        _write_weight_file(path, q, scales=scales, group_size=gs, num_groups=ng)

    # The initial LN is conventionally folded into embedding lookup.
    emb = _layer_norm(w["emb.weight"], w["blocks.0.ln0.weight"], w["blocks.0.ln0.bias"])
    save_matrix("embed_tokens", emb)
    save_matrix("lm_head", w["head.weight"])
    _write_weight_file(os.path.join(out, "final_norm_w.weights"), w["ln_out.weight"].astype(np.float32))
    _write_weight_file(os.path.join(out, "final_norm_b.weights"), w["ln_out.bias"].astype(np.float32))

    # Keep the large matrices FP16. The recurrent WKV and token-shift state
    # remain FP32 in the runtime; scalar/norm parameters also stay FP32.
    matrix_suffixes = ("receptance.weight", "key.weight", "value.weight", "output.weight",
                       ".w1", ".w2", ".a1", ".a2", ".v1", ".v2", ".g1", ".g2")
    for i in range(layers):
        prefix = f"blocks.{i}."
        for key, arr in w.items():
            if not key.startswith(prefix) or key.startswith(prefix + "ln0"):
                continue
            name = key.replace(".", "_")
            # LoRA checkpoint matrices use [in, rank] then [rank, out], while
            # mollm's matmul consumes [out, in].
            if key.startswith(prefix + "att.") and key.endswith((".w1", ".w2", ".a1", ".a2", ".v1", ".v2", ".g1", ".g2")):
                arr = arr.T
            if key.endswith(matrix_suffixes):
                save_matrix(name, arr)
            else:
                _write_weight_file(os.path.join(out, f"{name}.weights"), arr.astype(np.float32))


def _weight(g: GraphBuilder, root: str, name: str, shape: tuple,
            fp32=False, quant="fp16") -> int:
    return g.weight(os.path.join(root, f"{name}.weights"), shape,
                    Precision.FP32 if fp32 else _matrix_precision(quant, name, shape[1]))


def _scalar_weight(g: GraphBuilder, root: str, name: str, shape: tuple) -> int:
    return _weight(g, root, name, shape, fp32=True)


def _mix(g: GraphBuilder, x: int, sx: int, mix: int) -> int:
    return g.rwkv_mix(x, sx, mix)


def _lora(g: GraphBuilder, x: int, w1: int, w2: int, activation=None) -> int:
    y = g.matmul(x, w1)
    if activation == "tanh":
        y = g.tanh(y)
    elif activation == "sigmoid_exact":
        y = g.sigmoid_exact(y)
    return g.matmul(y, w2)


def _build_attention_block(g: GraphBuilder, root: str, weights: dict[str, np.ndarray],
                           layer: int, x: int, state: int, att_shift: int,
                           v_first, hidden: int, heads: int, head_size: int,
                           seq_len: int, quant: str) -> tuple[int, int]:
    block = f"blocks_{layer}"
    attention = f"{block}_att"

    def scalar(name: str) -> int:
        return _scalar_weight(g, root, f"{attention}_{name}", (hidden,))

    def linear(name: str, source: int) -> int:
        array = weights[f"blocks.{layer}.att.{name}.weight"]
        weight = _weight(g, root, f"{attention}_{name}_weight",
                         tuple(array.shape), quant=quant)
        return g.matmul(source, weight)

    # .pth LoRA matrices are [hidden, rank] / [rank, hidden], unlike Linear weights.
    def lora(name: str, source: int, activation=None) -> int:
        first = weights[f"blocks.{layer}.att.{name}1"]
        second = weights[f"blocks.{layer}.att.{name}2"]
        w1 = _weight(g, root, f"{attention}_{name}1", tuple(first.shape[::-1]),
                     quant=quant)
        w2 = _weight(g, root, f"{attention}_{name}2", tuple(second.shape[::-1]),
                     quant=quant)
        return _lora(g, source, w1, w2, activation)

    ln_weight = _scalar_weight(g, root, f"{block}_ln1_weight", (hidden,))
    ln_bias = _scalar_weight(g, root, f"{block}_ln1_bias", (hidden,))
    normalized = g.layer_norm(x, ln_weight, ln_bias)
    shifted = g.rwkv_token_shift(normalized, att_shift, hidden, seq_len)

    mix_weights = [scalar(name) for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")]
    xr, xw, xk, xv, xa, xg = (_mix(g, normalized, shifted, mix) for mix in mix_weights)
    receptance = linear("receptance", xr)
    key = linear("key", xk)
    value = linear("value", xv)

    w_delta = lora("w", xw, "tanh")
    a_delta = lora("a", xa)
    gate = lora("g", xg, "sigmoid_exact")
    decay = g.exp_exact(g.scalar_mul(g.sigmoid_exact(g.add(w_delta, scalar("w0"))), -0.606531))
    alpha = g.sigmoid_exact(g.add(a_delta, scalar("a0")))
    normalized_key = g.rwkv_l2_norm(g.mul(key, scalar("k_k")), heads, head_size, 1e-6)
    rwkv_key = g.mul(
        key,
        g.scalar_add(g.mul(g.scalar_add(alpha, -1.0), scalar("k_a")), 1.0),
    )

    if layer == 0:
        v_first = value
        rwkv_value = value
    else:
        v_delta = lora("v", xv)
        value_mix = g.sigmoid_exact(g.add(v_delta, scalar("v0")))
        rwkv_value = g.add(
            value,
            g.mul(g.add(v_first, g.scalar_mul(value, -1.0)), value_mix),
        )

    raw = g.rwkv7_core(
        receptance,
        decay,
        rwkv_key,
        rwkv_value,
        g.scalar_mul(normalized_key, -1.0),
        g.mul(normalized_key, alpha),
        state,
        heads,
        head_size,
        seq_len,
    )
    attention_out = g.rwkv_post(
        raw,
        receptance,
        rwkv_key,
        rwkv_value,
        scalar("r_k"),
        scalar("ln_x_weight"),
        scalar("ln_x_bias"),
        gate,
        heads,
        head_size,
    )
    output_array = weights[f"blocks.{layer}.att.output.weight"]
    output_weight = _weight(g, root, f"{attention}_output_weight",
                            tuple(output_array.shape), quant=quant)
    return g.add(x, g.matmul(attention_out, output_weight)), v_first


def _build_ffn_block(g: GraphBuilder, root: str, weights: dict[str, np.ndarray],
                     layer: int, x: int, ffn_shift: int, hidden: int,
                     seq_len: int, quant: str) -> int:
    block = f"blocks_{layer}"
    ffn = f"{block}_ffn"
    ln_weight = _scalar_weight(g, root, f"{block}_ln2_weight", (hidden,))
    ln_bias = _scalar_weight(g, root, f"{block}_ln2_bias", (hidden,))
    normalized = g.layer_norm(x, ln_weight, ln_bias)
    shifted = g.rwkv_token_shift(normalized, ffn_shift, hidden, seq_len)
    mixed_key = _mix(g, normalized, shifted, _scalar_weight(g, root, f"{ffn}_x_k", (hidden,)))

    key_array = weights[f"blocks.{layer}.ffn.key.weight"]
    value_array = weights[f"blocks.{layer}.ffn.value.weight"]
    key_weight = _weight(g, root, f"{ffn}_key_weight", tuple(key_array.shape), quant=quant)
    value_weight = _weight(g, root, f"{ffn}_value_weight", tuple(value_array.shape), quant=quant)
    key = g.matmul(mixed_key, key_weight, activation=Activation.RELU_SQUARED)
    return g.add(x, g.gemv_sparse_a(key, value_weight))


def build_graph(root: str, w: dict[str, np.ndarray], seq_len: int, prefill: bool,
                quant: str = "fp16") -> GraphBuilder:
    layers, vocab, hidden, heads, hs = _info(w)
    g = GraphBuilder()
    g.set_model_config(hidden_size=hidden, num_layers=layers, vocab_size=vocab,
                       model_type="rwkv7", rope_dim=0, rope_theta=0)
    g.weight(os.path.join(root, "embed_tokens.weights"), (vocab, hidden), Precision.FP16)
    g.weight(os.path.join(root, "lm_head.weights"), (vocab, hidden),
             _matrix_precision(quant, "lm_head", hidden))
    dyn = (DimExpr.const(), DimExpr.seq(), DimExpr.const(), DimExpr.const()) if prefill else None
    x = g.input("hidden", (hidden, seq_len), dynamic=dyn)
    states = []
    for i in range(layers):
        # Match llama.cpp and the reference recurrence: WKV state remains FP32.
        # This also avoids a full matrix conversion around every token.
        states.append((
            g.input(f"rwkv_state{i}", (hs, hs, heads), Precision.FP32),
            g.input(f"rwkv_att_shift{i}", (hidden,), Precision.FP16),
            g.input(f"rwkv_ffn_shift{i}", (hidden,), Precision.FP16),
        ))

    v_first = None
    for i in range(layers):
        state, att_shift, ffn_shift = states[i]
        x, v_first = _build_attention_block(
            g, root, w, i, x, state, att_shift, v_first, hidden, heads, hs, seq_len, quant,
        )
        x = _build_ffn_block(g, root, w, i, x, ffn_shift, hidden, seq_len, quant)
    x = g.layer_norm(
        x,
        _scalar_weight(g, root, "final_norm_w", (hidden,)),
        _scalar_weight(g, root, "final_norm_b", (hidden,)),
    )
    return g


def convert_rwkv7(checkpoint: str, output_path: str, prefill_seq_len: int = 256,
                  tokenizer_path: str = "", quant: str = "fp16") -> None:
    path = Path(checkpoint)
    weights = _torch_load(path)
    quant = _canonical_quant(quant)
    layers, vocab, hidden, heads, head_size = _info(weights)
    metadata = {
        "model_name": path.stem,
        "architecture": "rwkv7",
        "num_layers": layers,
        "hidden_size": hidden,
        "num_heads": heads,
        "head_dim": head_size,
        "vocab_size": vocab,
        "prefill_seq_len": prefill_seq_len,
        "quantization": quant,
        "rwkv_chat_template": "rwkv_legacy",
        "experimental": True,
    }
    tmp = tempfile.mkdtemp(prefix="mollm_rwkv7_")
    try:
        export_weights(weights, tmp, quant)
        prefill_graph = build_graph(".", weights, prefill_seq_len, True, quant)
        decode_graph = build_graph(".", weights, 1, False, quant)
        save_package(output_path, prefill_graph, decode_graph, tmp, metadata,
                     tokenizer_path=tokenizer_path)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="convert an RWKV v7 .pth checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--prefill-seq-len", type=int, default=256)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--quant", default="fp16",
                        choices=("fp16", "w8", "w8pc", "w4", "w4mixg128"))
    args = parser.parse_args()
    convert_rwkv7(args.checkpoint, args.output, args.prefill_seq_len,
                  args.tokenizer, args.quant)
