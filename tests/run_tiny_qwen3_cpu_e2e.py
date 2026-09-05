#!/usr/bin/env python3
"""Generate tiny Qwen3 packages and run the CPU engine against both.

This fixture deliberately exercises the public converter, package loader and
runtime together.  It stays tiny enough for CTest while retaining the INT4
BG32 format used by real W4G32 packages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


PACKAGE_WEIGHTS_OFFSET = 88
WEIGHT_HEADER_SIZE = 88
WEIGHT_INT4_BG32_FLAG = 1 << 2


def write_safetensors(path: Path, tensors: dict[str, object]) -> None:
    import numpy as np

    header: dict[str, object] = {}
    parts: list[bytes] = []
    offset = 0
    for name, value in tensors.items():
        array = np.ascontiguousarray(value)
        if array.dtype == np.float16:
            dtype = "F16"
        elif array.dtype == np.float32:
            dtype = "F32"
        else:
            raise TypeError(f"unsupported fixture dtype {array.dtype}")
        payload = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        parts.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(parts))


def build_qwen3_fixture_model(model_dir: Path) -> None:
    import numpy as np

    hidden = 32
    heads = 4
    head_dim = 8
    kv_heads = 1
    intermediate = 64
    vocab = 32
    rng = np.random.default_rng(20260729)

    def matrix(rows: int, cols: int) -> object:
        return rng.uniform(-0.1, 0.1, (rows, cols)).astype(np.float16)

    config = {
        "model_type": "qwen3",
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.embed_tokens.weight": matrix(vocab, hidden),
        "lm_head.weight": matrix(vocab, hidden),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": matrix(heads * head_dim, hidden),
        "model.layers.0.self_attn.k_proj.weight": matrix(kv_heads * head_dim, hidden),
        "model.layers.0.self_attn.v_proj.weight": matrix(kv_heads * head_dim, hidden),
        "model.layers.0.self_attn.q_norm.weight": np.ones(head_dim, dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.ones(head_dim, dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": matrix(hidden, hidden),
        "model.layers.0.mlp.gate_proj.weight": matrix(intermediate, hidden),
        "model.layers.0.mlp.up_proj.weight": matrix(intermediate, hidden),
        "model.layers.0.mlp.down_proj.weight": matrix(hidden, intermediate),
    }
    write_safetensors(model_dir / "model.safetensors", tensors)


def build_qwen35_fixture_model(
        model_dir: Path, force_accept: bool = False) -> None:
    import numpy as np

    hidden = 32
    head_dim = 8
    linear_heads = 2
    linear_value_heads = 4
    linear_k_dim = 8
    linear_v_dim = 8
    qkv_total = linear_heads * linear_k_dim * 2 + linear_value_heads * linear_v_dim
    intermediate = 64
    vocab = 1 if force_accept else 32
    rng = np.random.default_rng(20260730)

    def matrix(rows: int, cols: int) -> object:
        return rng.uniform(-0.1, 0.1, (rows, cols)).astype(np.float16)

    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
            },
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": head_dim,
            "linear_num_key_heads": linear_heads,
            "linear_key_head_dim": linear_k_dim,
            "linear_value_head_dim": linear_v_dim,
            "linear_num_value_heads": linear_value_heads,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "mtp_num_hidden_layers": 1,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    prefix = "model.language_model.layers.0"
    tensors = {
        "model.language_model.embed_tokens.weight": matrix(vocab, hidden),
        "lm_head.weight": (
            np.zeros((vocab, hidden), dtype=np.float16)
            if force_accept else matrix(vocab, hidden)),
        "model.language_model.norm.weight": np.zeros(hidden, dtype=np.float32),
        f"{prefix}.input_layernorm.weight": np.zeros(hidden, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.zeros(hidden, dtype=np.float32),
        f"{prefix}.linear_attn.in_proj_qkv.weight": matrix(qkv_total, hidden),
        f"{prefix}.linear_attn.in_proj_a.weight": matrix(linear_value_heads, hidden),
        f"{prefix}.linear_attn.in_proj_b.weight": matrix(linear_value_heads, hidden),
        f"{prefix}.linear_attn.in_proj_z.weight": matrix(linear_value_heads * linear_v_dim, hidden),
        f"{prefix}.linear_attn.A_log": np.zeros(linear_value_heads, dtype=np.float32),
        f"{prefix}.linear_attn.dt_bias": np.zeros(linear_value_heads, dtype=np.float32),
        f"{prefix}.linear_attn.conv1d.weight": rng.uniform(
            -0.05, 0.05, (qkv_total, 1, 4)).astype(np.float32),
        f"{prefix}.linear_attn.norm.weight": np.ones(linear_v_dim, dtype=np.float32),
        f"{prefix}.linear_attn.out_proj.weight": matrix(
            hidden, linear_value_heads * linear_v_dim),
        f"{prefix}.mlp.gate_proj.weight": matrix(intermediate, hidden),
        f"{prefix}.mlp.up_proj.weight": matrix(intermediate, hidden),
        f"{prefix}.mlp.down_proj.weight": matrix(hidden, intermediate),
        "mtp.pre_fc_norm_embedding.weight": np.zeros(hidden, dtype=np.float32),
        "mtp.pre_fc_norm_hidden.weight": np.zeros(hidden, dtype=np.float32),
        "mtp.fc.weight": matrix(hidden, 2 * hidden),
        "mtp.layers.0.input_layernorm.weight": np.zeros(hidden, dtype=np.float32),
        "mtp.layers.0.post_attention_layernorm.weight": np.zeros(hidden, dtype=np.float32),
        "mtp.layers.0.self_attn.q_proj.weight": matrix(2 * 2 * head_dim, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": matrix(head_dim, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": matrix(head_dim, hidden),
        "mtp.layers.0.self_attn.q_norm.weight": np.zeros(head_dim, dtype=np.float32),
        "mtp.layers.0.self_attn.k_norm.weight": np.zeros(head_dim, dtype=np.float32),
        "mtp.layers.0.self_attn.o_proj.weight": matrix(hidden, 2 * head_dim),
        "mtp.layers.0.mlp.gate_proj.weight": matrix(intermediate, hidden),
        "mtp.layers.0.mlp.up_proj.weight": matrix(intermediate, hidden),
        "mtp.layers.0.mlp.down_proj.weight": matrix(hidden, intermediate),
        "mtp.norm.weight": np.zeros(hidden, dtype=np.float32),
    }
    write_safetensors(model_dir / "model.safetensors", tensors)


def assert_has_bg32_weight(package: Path) -> None:
    """Ensure the W4 E2E package reaches the native BG32 load path."""
    data = package.read_bytes()
    meta_offset, meta_length = struct.unpack_from("<QQ", data, 8)
    weights_offset, _ = struct.unpack_from("<QQ", data, PACKAGE_WEIGHTS_OFFSET)
    metadata = json.loads(data[meta_offset:meta_offset + meta_length])
    for relative_offset, size in metadata["weights"].values():
        if size < WEIGHT_HEADER_SIZE:
            continue
        base = weights_offset + relative_offset
        flags, _, precision = struct.unpack_from("<III", data, base + 4)
        if precision == 3 and flags & WEIGHT_INT4_BG32_FLAG:
            return
    raise AssertionError("tiny W4G32 package contains no native BG32 weight")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: run_tiny_qwen3_cpu_e2e.py <converter.py> <quantizer> <runner>",
              file=sys.stderr)
        return 2
    converter, quantizer, runner = map(Path, sys.argv[1:])
    if not converter.is_file() or not quantizer.is_file() or not runner.is_file():
        print("tiny e2e prerequisites are missing", file=sys.stderr)
        return 2
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("SKIP: numpy is required to build the tiny converter fixture")
        return 77

    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_qwen3_"))
    try:
        qwen3_dir = temp_dir / "qwen3"
        qwen3_dir.mkdir()
        build_qwen3_fixture_model(qwen3_dir)
        qwen35_reject_dir = temp_dir / "qwen35_reject"
        qwen35_reject_dir.mkdir()
        build_qwen35_fixture_model(qwen35_reject_dir)
        qwen35_accept_dir = temp_dir / "qwen35_accept"
        qwen35_accept_dir.mkdir()
        build_qwen35_fixture_model(qwen35_accept_dir, force_accept=True)
        fp16 = temp_dir / "tiny_fp16.mollm"
        w4 = temp_dir / "tiny_w4g32.mollm"
        qwen35_reject = temp_dir / "tiny_qwen35_reject_fp16.mollm"
        qwen35_accept = temp_dir / "tiny_qwen35_accept_fp16.mollm"
        environment = os.environ.copy()
        environment["MOLLM_QUANT_HELPER"] = str(quantizer)
        environment["MOLLM_QUANT_THREADS"] = "1"
        for model_dir, package, quant in (
                (qwen3_dir, fp16, "fp16"),
                (qwen3_dir, w4, "w4g32"),
                (qwen35_reject_dir, qwen35_reject, "fp16"),
                (qwen35_accept_dir, qwen35_accept, "fp16")):
            subprocess.run(
                [sys.executable, str(converter), str(model_dir), str(package), quant],
                check=True, env=environment, cwd=str(converter.parent.parent))
        assert_has_bg32_weight(w4)
        subprocess.run(
            [str(runner), str(fp16), str(w4), str(qwen35_reject),
             str(qwen35_accept)], check=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
