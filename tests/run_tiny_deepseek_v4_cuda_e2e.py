#!/usr/bin/env python3
"""Build a native DeepSeek-V4 checkpoint and test its converted package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def write_native_safetensors(path: Path, tensors: dict[str, tuple]) -> None:
    """Write the native dtype labels used by DeepSeek-V4 safetensors."""
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, (dtype, shape, payload) in tensors.items():
        raw = bytes(payload)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def build_checkpoint(model_dir: Path) -> None:
    import numpy as np

    hidden = 32
    intermediate = 32
    experts = 4
    top_k = 2
    vocab = 8
    heads = 1
    head_dim = 32
    q_rank = 32
    o_rank = 8
    o_groups = 4
    index_heads = 1
    index_dim = 32
    ratio = 4
    hc_mult = 3
    wide = hidden * hc_mult
    hc_mix = (2 + hc_mult) * hc_mult
    rng = np.random.default_rng(20260804)

    config = {
        "model_type": "deepseek_v4",
        "hidden_size": hidden,
        "num_hidden_layers": 2,
        "vocab_size": vocab,
        "n_routed_experts": experts,
        "moe_intermediate_size": intermediate,
        "num_experts_per_tok": top_k,
        "num_hash_layers": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 0.7,
        "swiglu_limit": 0.5,
        "num_attention_heads": heads,
        "head_dim": head_dim,
        "q_lora_rank": q_rank,
        "o_lora_rank": o_rank,
        "o_groups": o_groups,
        "sliding_window": 3,
        "compress_ratios": [ratio, 0],
        "qk_rope_head_dim": 0,
        "index_topk": top_k,
        "index_n_heads": index_heads,
        "index_head_dim": index_dim,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 8,
            "factor": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "hc_mult": hc_mult,
        "hc_sinkhorn_iters": 5,
        "hc_eps": 1e-6,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text(
        json.dumps({
            "model": {
                "type": "BPE",
                "vocab": {chr(ord("a") + token): token
                          for token in range(vocab)},
                "merges": [],
            },
            "added_tokens": [],
        }),
        encoding="utf-8",
    )

    tensors: dict[str, tuple] = {}

    def add(name: str, values, dtype: str | None = None,
            logical_shape: tuple[int, ...] | None = None) -> None:
        array = np.ascontiguousarray(values)
        label = dtype
        if label is None:
            label = {
                np.dtype(np.float16): "F16",
                np.dtype(np.float32): "F32",
                np.dtype(np.int32): "I32",
            }.get(array.dtype)
        if label is None:
            raise TypeError(f"unsupported fixture dtype: {array.dtype}")
        tensors[name] = (
            label, logical_shape or tuple(array.shape), array.tobytes())

    def add_bf16(name: str, values) -> None:
        fp32 = np.ascontiguousarray(values, dtype=np.float32)
        bf16 = (fp32.view(np.uint32) >> np.uint32(16)).astype("<u2")
        add(name, bf16, "BF16", tuple(fp32.shape))

    def matrix(rows: int, columns: int, scale: float = 0.08):
        return rng.uniform(
            -scale, scale, (rows, columns)).astype(np.float32)

    fp8_codes = np.array([
        0x00, 0x10, 0x18, 0x20, 0x28,
        0x90, 0x98, 0xA0, 0xA8,
    ], dtype=np.uint8)

    def add_fp8(name: str, rows: int, columns: int) -> None:
        values = rng.choice(fp8_codes, size=(rows, columns))
        scale_shape = ((rows + 127) // 128, (columns + 127) // 128)
        scales = np.full(scale_shape, 124, dtype=np.uint8)
        add(name, values, "F8_E4M3")
        add(name.removesuffix(".weight") + ".scale",
            scales, "F8_E8M0")

    mxfp4_codes = np.array([0x00, 0x01, 0x02, 0x08, 0x09, 0x0A],
                            dtype=np.uint8)

    def add_mxfp4(name: str, rows: int, columns: int) -> None:
        if columns % 32:
            raise ValueError("MXFP4 logical width must be divisible by 32")
        low = rng.choice(mxfp4_codes, size=(rows, columns // 2))
        high = rng.choice(mxfp4_codes, size=(rows, columns // 2))
        packed = low | (high << np.uint8(4))
        scales = np.full((rows, columns // 32), 124, dtype=np.uint8)
        add(name, packed, "I8")
        add(name.removesuffix(".weight") + ".scale",
            scales, "F8_E8M0")

    add_bf16("embed.weight", matrix(vocab, hidden))
    add_bf16("head.weight", matrix(vocab, hidden))

    for layer_index, layer_ratio in enumerate((ratio, 0)):
        layer = f"layers.{layer_index}"
        add_bf16(f"{layer}.attn_norm.weight", np.ones(hidden))
        add_bf16(f"{layer}.ffn_norm.weight", np.ones(hidden))
        add(f"{layer}.ffn.gate.weight",
            rng.uniform(-0.05, 0.05,
                        (experts, hidden)).astype(np.float32))
        if layer_index == 0:
            # Native layout is [vocab, top_k]; the converter exposes
            # [top_k, vocab].
            add(f"{layer}.ffn.gate.tid2eid", np.array([
                [3, 1], [2, 0], [1, 3], [3, 0],
                [0, 2], [2, 1], [1, 0], [0, 3],
            ], dtype=np.int32))
        else:
            add_bf16(f"{layer}.ffn.gate.bias",
                     np.array([0.03, -0.07, 0.11, -0.02]))
        for expert in range(experts):
            prefix = f"{layer}.ffn.experts.{expert}"
            add_mxfp4(f"{prefix}.w1.weight", intermediate, hidden)
            add_mxfp4(f"{prefix}.w3.weight", intermediate, hidden)
            add_mxfp4(f"{prefix}.w2.weight", hidden, intermediate)
        shared = f"{layer}.ffn.shared_experts"
        add_fp8(f"{shared}.w1.weight", intermediate, hidden)
        add_fp8(f"{shared}.w3.weight", intermediate, hidden)
        add_fp8(f"{shared}.w2.weight", hidden, intermediate)

        attention = f"{layer}.attn"
        add_fp8(f"{attention}.wq_a.weight", q_rank, hidden)
        add_bf16(f"{attention}.q_norm.weight", np.ones(q_rank))
        add_fp8(f"{attention}.wq_b.weight", heads * head_dim, q_rank)
        add_fp8(f"{attention}.wkv.weight", head_dim, hidden)
        add_bf16(f"{attention}.kv_norm.weight", np.ones(head_dim))
        add_bf16(f"{attention}.attn_sink", np.array([-0.125]))
        add_fp8(f"{attention}.wo_a.weight", o_groups * o_rank,
                heads * head_dim // o_groups)
        add_fp8(f"{attention}.wo_b.weight", hidden, o_groups * o_rank)

        if layer_ratio:
            projected = 2 * head_dim
            add_bf16(f"{attention}.compressor.wkv.weight",
                     rng.uniform(-0.04, 0.04, (projected, hidden)))
            add_bf16(f"{attention}.compressor.wgate.weight",
                     rng.uniform(-0.02, 0.02, (projected, hidden)))
            # APE is physically [ratio, projected], matching the native
            # checkpoint.
            add_bf16(f"{attention}.compressor.ape",
                     rng.uniform(-0.02, 0.02,
                                 (layer_ratio, projected)))
            add_bf16(f"{attention}.compressor.norm.weight",
                     np.ones(head_dim))

            indexer = f"{attention}.indexer"
            add_fp8(f"{indexer}.wq_b.weight",
                    index_heads * index_dim, q_rank)
            add_bf16(f"{indexer}.weights_proj.weight",
                     matrix(index_heads, hidden, 0.02))
            index_projected = 2 * index_dim
            add_bf16(f"{indexer}.compressor.wkv.weight",
                     rng.uniform(-0.04, 0.04,
                                 (index_projected, hidden)))
            add_bf16(f"{indexer}.compressor.wgate.weight",
                     rng.uniform(-0.02, 0.02,
                                 (index_projected, hidden)))
            add_bf16(f"{indexer}.compressor.ape",
                     rng.uniform(-0.02, 0.02,
                                 (layer_ratio, index_projected)))
            add_bf16(f"{indexer}.compressor.norm.weight",
                     np.ones(index_dim))

        for kind in ("attn", "ffn"):
            prefix = f"{layer}.hc_{kind}"
            add_bf16(f"{prefix}_fn",
                     rng.uniform(-0.08, 0.08, (hc_mix, wide)))
            add_bf16(f"{prefix}_scale",
                     np.array([0.75, 1.25, -0.55]))
            add_bf16(f"{prefix}_base",
                     rng.uniform(-0.08, 0.08, hc_mix))
    add_bf16("hc_head_fn", rng.uniform(-0.08, 0.08, (hc_mult, wide)))
    add_bf16("hc_head_scale", np.array([0.9]))
    add_bf16("hc_head_base", rng.uniform(-0.08, 0.08, hc_mult))
    add_bf16("norm.weight", np.ones(hidden))

    write_native_safetensors(model_dir / "model.safetensors", tensors)


def validate_package(package: Path) -> None:
    data = package.read_bytes()
    metadata_offset, metadata_length = struct.unpack_from("<QQ", data, 8)
    weights_offset, _ = struct.unpack_from("<QQ", data, 88)
    metadata = json.loads(
        data[metadata_offset:metadata_offset + metadata_length])
    if metadata.get("architecture") != "deepseek-v4":
        raise AssertionError("converted package lost DeepSeek-V4 architecture")
    if metadata.get("quantization") != "native-fp8-mxfp4":
        raise AssertionError("converted package lost native quantization")
    if metadata.get("num_layers") != 2:
        raise AssertionError("converted package lost the two-layer fixture")
    expected = {
        "./embed_tokens.weights": 1,
        "./layer_0_attn_wq_a_weight.weights": 4,
        "./layer_0_attn_q_norm_weight.weights": 0,
        "./layer_0_experts_gate_up.weights": 5,
        "./layer_0_tid2eid.weights": 6,
        "./layer_1_router_bias.weights": 0,
        "./layer_1_experts_down.weights": 5,
    }
    for name, precision in expected.items():
        relative_offset, size = metadata["weights"][name]
        if size < 88:
            raise AssertionError(f"truncated packaged weight: {name}")
        actual = struct.unpack_from(
            "<I", data, weights_offset + relative_offset + 12)[0]
        if actual != precision:
            raise AssertionError(
                f"{name} precision {actual} != expected {precision}")


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <models/deepseek_v4.py> <e2e-runner>",
            file=sys.stderr,
        )
        return 2
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("SKIP: numpy is required for the DeepSeek-V4 converter E2E")
        return 77

    converter = Path(sys.argv[1]).resolve()
    runner = Path(sys.argv[2]).resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_dsv4_"))
    try:
        model_dir = temp_dir / "checkpoint"
        model_dir.mkdir()
        build_checkpoint(model_dir)
        package = temp_dir / "tiny-deepseek-v4-native.mollm"
        subprocess.run(
            [
                sys.executable,
                str(converter),
                str(model_dir),
                str(package),
                "--prefill-seq-len",
                "4",
                "--n-ctx",
                "8",
            ],
            check=True,
            cwd=str(converter.parent.parent),
        )
        validate_package(package)
        environment = os.environ.copy()
        environment["MOLLM_CUDA_PROFILE"] = "1"
        completed = subprocess.run(
            [str(runner), str(package)],
            check=False,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode == 77:
            return 77
        completed.check_returncode()
        if "  fallback " in completed.stderr:
            raise AssertionError(
                "tiny DeepSeek-V4 CUDA converter E2E used an operator fallback")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
