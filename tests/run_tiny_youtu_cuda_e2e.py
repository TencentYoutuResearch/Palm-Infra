#!/usr/bin/env python3
"""Convert a tiny BF16 Youtu MLA checkpoint and compare CPU/CUDA."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def write_safetensors(path: Path, tensors: dict[str, tuple]) -> None:
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
    heads = 2
    kv_rank = 32
    q_rank = 32
    qk_nope = 16
    qk_rope = 16
    v_head = 16
    intermediate = 32
    vocab = 32
    rng = np.random.default_rng(20260805)

    config = {
        "model_type": "youtu",
        "_name_or_path": "tiny-youtu-mla",
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "head_dim": qk_nope + qk_rope,
        "kv_lora_rank": kv_rank,
        "q_lora_rank": q_rank,
        "qk_nope_head_dim": qk_nope,
        "qk_rope_head_dim": qk_rope,
        "v_head_dim": v_head,
        "intermediate_size": intermediate,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text(
        json.dumps({
            "model": {
                "type": "BPE",
                "vocab": {chr(ord("a") + token): token
                          for token in range(26)} | {
                              f"token_{token}": token
                              for token in range(26, vocab)
                          },
                "merges": [],
            },
            "added_tokens": [],
        }),
        encoding="utf-8",
    )

    tensors: dict[str, tuple] = {}

    def add_bf16(name: str, values) -> None:
        fp32 = np.ascontiguousarray(values, dtype=np.float32)
        bf16 = (fp32.view(np.uint32) >> np.uint32(16)).astype("<u2")
        tensors[name] = ("BF16", tuple(fp32.shape), bf16.tobytes())

    def matrix(rows: int, columns: int, scale: float = 0.08):
        return rng.uniform(-scale, scale, (rows, columns))

    add_bf16("model.embed_tokens.weight", matrix(vocab, hidden))
    add_bf16("lm_head.weight", matrix(vocab, hidden, 0.06))
    add_bf16("model.norm.weight", np.ones(hidden))

    layer = "model.layers.0"
    attention = f"{layer}.self_attn"
    add_bf16(f"{layer}.input_layernorm.weight", np.ones(hidden))
    add_bf16(f"{attention}.q_a_proj.weight", matrix(q_rank, hidden))
    add_bf16(f"{attention}.q_a_layernorm.weight", np.ones(q_rank))
    add_bf16(
        f"{attention}.q_b_proj.weight",
        matrix(heads * (qk_nope + qk_rope), q_rank),
    )
    add_bf16(
        f"{attention}.kv_a_proj_with_mqa.weight",
        matrix(kv_rank + qk_rope, hidden),
    )
    add_bf16(f"{attention}.kv_a_layernorm.weight", np.ones(kv_rank))
    add_bf16(
        f"{attention}.kv_b_proj.weight",
        matrix(heads * (qk_nope + v_head), kv_rank),
    )
    add_bf16(
        f"{attention}.o_proj.weight", matrix(hidden, heads * v_head))
    add_bf16(f"{layer}.post_attention_layernorm.weight", np.ones(hidden))
    add_bf16(f"{layer}.mlp.gate_proj.weight",
             matrix(intermediate, hidden))
    add_bf16(f"{layer}.mlp.up_proj.weight",
             matrix(intermediate, hidden))
    add_bf16(f"{layer}.mlp.down_proj.weight",
             matrix(hidden, intermediate))

    write_safetensors(model_dir / "model.safetensors", tensors)


def validate_package(package: Path) -> None:
    data = package.read_bytes()
    metadata_offset, metadata_length = struct.unpack_from("<QQ", data, 8)
    weights_offset, _ = struct.unpack_from("<QQ", data, 88)
    metadata = json.loads(
        data[metadata_offset:metadata_offset + metadata_length])
    if metadata.get("architecture") != "mla":
        raise AssertionError("converted package lost MLA architecture")
    if metadata.get("quantization") != "w4g32":
        raise AssertionError("converted package lost W4G32 quantization")
    expected = {
        "./embed_tokens.weights": (1, 0),
        "./lm_head.weights": (3, 1 << 2),
        "./model_layers_0_self_attn_q_a_proj_weight.weights": (3, 1 << 2),
        "./model_layers_0_input_layernorm_weight.weights": (0, 0),
    }
    for name, (precision, required_flags) in expected.items():
        relative_offset, size = metadata["weights"][name]
        if size < 88:
            raise AssertionError(f"truncated packaged weight: {name}")
        flags, _, actual_precision = struct.unpack_from(
            "<III", data, weights_offset + relative_offset + 4)
        if actual_precision != precision or flags & required_flags != required_flags:
            raise AssertionError(
                f"unexpected header for {name}: precision={actual_precision}, "
                f"flags=0x{flags:x}")


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} <models/converter.py> <quantizer> "
            "<e2e-runner>",
            file=sys.stderr,
        )
        return 2
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("SKIP: numpy is required for the Youtu converter E2E")
        return 77

    converter, quantizer, runner = (
        Path(argument).resolve() for argument in sys.argv[1:])
    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_youtu_"))
    try:
        model_dir = temp_dir / "checkpoint"
        model_dir.mkdir()
        build_checkpoint(model_dir)
        package = temp_dir / "tiny-youtu-w4g32.mollm"
        environment = os.environ.copy()
        environment["MOLLM_QUANT_HELPER"] = str(quantizer)
        environment["MOLLM_QUANT_THREADS"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(converter),
                str(model_dir),
                str(package),
                "w4g32",
            ],
            check=True,
            env=environment,
            cwd=str(converter.parent.parent),
        )
        validate_package(package)
        # The AVX-512 VNNI W4 GEMM additionally quantizes activations to Q8.
        # Use the non-lossy scalar W4 path as the CUDA correctness oracle.
        environment["MOLLM_X86_ISA"] = "scalar"
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
                "tiny Youtu CUDA converter E2E used an operator fallback")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
