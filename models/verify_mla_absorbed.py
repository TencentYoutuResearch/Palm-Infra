#!/usr/bin/env python3
"""Compare Hy4 MLA absorption using actual safetensors and GGUF K/V weights.

Usage: python3 models/verify_mla_absorbed.py CHECKPOINT.gguf REFERENCE_DIR
       [--layer 0] [--heads 0 1 17 63]

The reference computes explicit K/V from fused kv_b_proj; GGUF computes the
absorbed query and expands V from its split q8_0 tensors. The comparison uses
realistic activation magnitudes and several positions, so softmax scaling and
the K layout both affect the result.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from gguf_quants import dequantize_for_shape, ggml_type_info
from gguf_reader import open_gguf
from safetensors_stream import SafeTensorIndex, split_mla_kv_b


def reference_fused(index: SafeTensorIndex, name: str) -> np.ndarray:
    tensor = index.tensor(name)
    with open(tensor.source.path, "rb") as source:
        source.seek(tensor.source.offset)
        raw = source.read(tensor.source.size)
    if len(raw) != tensor.source.size:
        raise EOFError(name)
    if tensor.dtype == "BF16":
        bits = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
        values = bits.view(np.float32)
    elif tensor.dtype == "F16":
        values = np.frombuffer(raw, "<f2").astype(np.float32)
    else:
        raise ValueError(f"unexpected dtype {tensor.dtype} for {name}")
    return values.reshape(tensor.shape)


def gguf_weight(gguf, info) -> np.ndarray:
    entry = ggml_type_info(info.type_id)
    with open(gguf.path, "rb") as source:
        source.seek(gguf.data_start + info.offset)
        raw = source.read(entry.total_bytes(info.logical_shape))
    blocks = np.frombuffer(raw, np.uint8).reshape(-1, entry.block_bytes)
    return dequantize_for_shape(info.type_id, blocks, info.logical_shape)


def softmax_with_sink(scores: np.ndarray, sink: np.ndarray) -> np.ndarray:
    all_logits = np.concatenate([scores, sink[:, None]], axis=1)
    all_logits -= all_logits.max(axis=1, keepdims=True)
    weights = np.exp(all_logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights[:, :-1]


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) /
                 max(np.linalg.norm(a.astype(np.float64)), 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("reference_dir")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--heads", nargs="+", type=int, default=[0, 1, 17, 63])
    args = parser.parse_args()

    gguf, mapped = open_gguf(args.gguf)
    ref_dir = Path(args.reference_dir)
    cfg = json.loads((ref_dir / "config.json").read_text())
    heads = int(cfg["num_attention_heads"])
    rank = int(cfg["kv_lora_rank"])
    nope = int(cfg["qk_nope_head_dim"])
    rope = int(cfg["qk_rope_head_dim"])
    v_dim = int(cfg["v_head_dim"])
    selected = np.asarray(args.heads)
    if np.any((selected < 0) | (selected >= heads)):
        parser.error(f"heads must lie in 0..{heads - 1}")

    index = SafeTensorIndex(ref_dir)
    fused_name = f"model.layers.{args.layer}.self_attn.kv_b_proj.weight"
    fused = reference_fused(index, fused_name).reshape(heads, nope + v_dim, rank)
    ref_k = fused[:, :nope, :].transpose(0, 2, 1)
    ref_v = fused[:, nope:, :]

    # Verify the streamed FP16 splitter as well as the GGUF reader. This is
    # the path a safetensors conversion would actually write into a package.
    k_stream, v_stream = split_mla_kv_b(
        index, fused_name, num_heads=heads, nope_dim=nope,
        v_head_dim=v_dim, kv_lora_rank=rank)
    for label, streamed, wanted in (("k_b", k_stream, ref_k),
                                    ("v_b", v_stream, ref_v)):
        output = io.BytesIO()
        streamed._copy_ranges(output, streamed.data_ranges)
        actual = np.frombuffer(output.getvalue(), "<f2").reshape(streamed.logical_shape)
        if not np.array_equal(actual, wanted.astype("<f2")):
            raise AssertionError(f"streamed {label} layout differs from reference")

    prefix = f"model.layers.{args.layer}.self_attn"
    k_info = mapped.info(f"{prefix}.k_b.weight")
    v_info = mapped.info(f"{prefix}.v_b.weight")
    k_b = gguf_weight(gguf, k_info)[selected]
    v_b = gguf_weight(gguf, v_info)[selected]
    ref_k = ref_k[selected]
    ref_v = ref_v[selected]
    k_error = relative(ref_k, k_b)
    v_error = relative(ref_v, v_b)

    rng = np.random.default_rng(20260925)
    seq = 8
    activation_scale = 2.0
    latent = rng.standard_normal((seq, rank), dtype=np.float32) * activation_scale
    q_nope = rng.standard_normal((len(selected), nope), dtype=np.float32) * activation_scale
    q_rope = rng.standard_normal((len(selected), rope), dtype=np.float32) * activation_scale
    k_rope = rng.standard_normal((seq, rope), dtype=np.float32) * activation_scale
    sink = rng.standard_normal(len(selected), dtype=np.float32)

    # Reference: expand every historical key and value, then attend.
    key = np.einsum("sc,hcd->hsd", latent, ref_k, optimize=True)
    score_ref = np.einsum("hd,hsd->hs", q_nope, key, optimize=True)
    # GGUF: fold the K projection into the current query.
    q_abs = np.einsum("hd,hcd->hc", q_nope, k_b, optimize=True)
    score_gguf = np.einsum("hc,sc->hs", q_abs, latent, optimize=True)
    rope_score = np.einsum("hd,sd->hs", q_rope, k_rope, optimize=True)
    scale = 1 / np.sqrt(nope + rope)
    p_ref = softmax_with_sink((score_ref + rope_score) * scale, sink)
    p_gguf = softmax_with_sink((score_gguf + rope_score) * scale, sink)
    value_ref = np.einsum("sc,hdc->hsd", latent, ref_v, optimize=True)
    value_gguf = np.einsum("sc,hdc->hsd", latent, v_b, optimize=True)
    out_ref = np.einsum("hs,hsd->hd", p_ref, value_ref, optimize=True)
    out_gguf = np.einsum("hs,hsd->hd", p_gguf, value_gguf, optimize=True)

    print(f"layer {args.layer}, heads {list(selected)}, q8_0 vs reference")
    for label, error in (("k_b", k_error), ("v_b", v_error),
                         ("scores", relative(score_ref, score_gguf)),
                         ("softmax", relative(p_ref, p_gguf)),
                         ("values", relative(value_ref, value_gguf)),
                         ("output", relative(out_ref, out_gguf))):
        print(f"  {label:8} relative error {error:.4%}")
    if k_error > 0.02 or v_error > 0.02 or relative(out_ref, out_gguf) > 0.05:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
