#!/usr/bin/env python3
"""Build tiny Qwen3/Qwen3.5 W4G32 MoE packages and compare CPU/CUDA."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

from run_tiny_qwen3_cpu_e2e import assert_has_bg32_weight, write_safetensors


def assert_has_int8_weight(package: Path) -> None:
    data = package.read_bytes()
    meta_offset, meta_length = struct.unpack_from("<QQ", data, 8)
    weights_offset, _ = struct.unpack_from("<QQ", data, 88)
    metadata = json.loads(data[meta_offset:meta_offset + meta_length])
    for relative_offset, size in metadata["weights"].values():
        if size < 88:
            continue
        _, _, precision = struct.unpack_from(
            "<III", data, weights_offset + relative_offset + 4)
        if precision == 2:
            return
    raise AssertionError("tiny W8 package contains no INT8 weight")


def build_qwen3_fixture(model_dir: Path) -> None:
    import numpy as np

    hidden = 32
    heads = 4
    head_dim = 8
    kv_heads = 1
    intermediate = 64
    moe_intermediate = 32
    experts = 4
    vocab = 32
    rng = np.random.default_rng(20260731)

    def matrix(rows: int, cols: int) -> object:
        return rng.uniform(-0.08, 0.08, (rows, cols)).astype(np.float16)

    config = {
        "model_type": "qwen3_moe",
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "moe_intermediate_size": moe_intermediate,
        "num_experts": experts,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 0,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 0.75,
        "scoring_func": "sigmoid",
        "vocab_size": vocab,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": matrix(vocab, hidden),
        "lm_head.weight": matrix(vocab, hidden),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.self_attn.q_proj.weight": matrix(heads * head_dim, hidden),
        f"{prefix}.self_attn.k_proj.weight": matrix(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.v_proj.weight": matrix(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.q_norm.weight": np.ones(head_dim, dtype=np.float32),
        f"{prefix}.self_attn.k_norm.weight": np.ones(head_dim, dtype=np.float32),
        f"{prefix}.self_attn.o_proj.weight": matrix(hidden, hidden),
        f"{prefix}.mlp.gate.weight": matrix(experts, hidden),
        f"{prefix}.mlp.gate.e_score_correction_bias": np.array(
            [0.03, -0.07, 0.11, -0.02], dtype=np.float32),
        f"{prefix}.mlp.shared_experts.gate_proj.weight": matrix(
            moe_intermediate, hidden),
        f"{prefix}.mlp.shared_experts.up_proj.weight": matrix(
            moe_intermediate, hidden),
        f"{prefix}.mlp.shared_experts.down_proj.weight": matrix(
            hidden, moe_intermediate),
    }
    for expert in range(experts):
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        tensors[f"{expert_prefix}.gate_proj.weight"] = matrix(
            moe_intermediate, hidden)
        tensors[f"{expert_prefix}.up_proj.weight"] = matrix(
            moe_intermediate, hidden)
        tensors[f"{expert_prefix}.down_proj.weight"] = matrix(
            hidden, moe_intermediate)
    write_safetensors(model_dir / "model.safetensors", tensors)


def build_qwen35_fixture(model_dir: Path) -> None:
    import numpy as np

    hidden = 32
    head_dim = 8
    linear_heads = 2
    linear_value_heads = 4
    linear_k_dim = 8
    linear_v_dim = 8
    qkv_total = (
        linear_heads * linear_k_dim * 2 +
        linear_value_heads * linear_v_dim)
    moe_intermediate = 32
    experts = 4
    vocab = 32
    rng = np.random.default_rng(20260801)

    def matrix(rows: int, cols: int) -> object:
        return rng.uniform(-0.06, 0.06, (rows, cols)).astype(np.float16)

    config = {
        "model_type": "qwen3_5_moe",
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
            "moe_intermediate_size": moe_intermediate,
            "shared_expert_intermediate_size": moe_intermediate,
            "num_experts": experts,
            "num_experts_per_tok": 2,
            "vocab_size": vocab,
        },
    }
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    prefix = "model.language_model.layers.0"
    tensors = {
        "model.language_model.embed_tokens.weight": matrix(vocab, hidden),
        "lm_head.weight": matrix(vocab, hidden),
        "model.language_model.norm.weight": np.zeros(
            hidden, dtype=np.float32),
        f"{prefix}.input_layernorm.weight": np.zeros(
            hidden, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.zeros(
            hidden, dtype=np.float32),
        f"{prefix}.linear_attn.in_proj_qkv.weight": matrix(
            qkv_total, hidden),
        f"{prefix}.linear_attn.in_proj_a.weight": matrix(
            linear_value_heads, hidden),
        f"{prefix}.linear_attn.in_proj_b.weight": matrix(
            linear_value_heads, hidden),
        f"{prefix}.linear_attn.in_proj_z.weight": matrix(
            linear_value_heads * linear_v_dim, hidden),
        f"{prefix}.linear_attn.A_log": np.zeros(
            linear_value_heads, dtype=np.float32),
        f"{prefix}.linear_attn.dt_bias": np.zeros(
            linear_value_heads, dtype=np.float32),
        f"{prefix}.linear_attn.conv1d.weight": rng.uniform(
            -0.04, 0.04, (qkv_total, 1, 4)).astype(np.float32),
        f"{prefix}.linear_attn.norm.weight": np.ones(
            linear_v_dim, dtype=np.float32),
        f"{prefix}.linear_attn.out_proj.weight": matrix(
            hidden, linear_value_heads * linear_v_dim),
        f"{prefix}.mlp.gate.weight": matrix(experts, hidden),
        f"{prefix}.mlp.experts.gate_up_proj": matrix(
            experts * 2 * moe_intermediate, hidden).reshape(
                experts, 2 * moe_intermediate, hidden),
        f"{prefix}.mlp.experts.down_proj": matrix(
            experts * hidden, moe_intermediate).reshape(
                experts, hidden, moe_intermediate),
        f"{prefix}.mlp.shared_expert.gate_proj.weight": matrix(
            moe_intermediate, hidden),
        f"{prefix}.mlp.shared_expert.up_proj.weight": matrix(
            moe_intermediate, hidden),
        f"{prefix}.mlp.shared_expert.down_proj.weight": matrix(
            hidden, moe_intermediate),
        f"{prefix}.mlp.shared_expert_gate.weight": matrix(1, hidden),
    }
    write_safetensors(model_dir / "model.safetensors", tensors)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: run_tiny_qwen3_moe_cuda_e2e.py "
              "<converter.py> <quantizer> <runner>", file=sys.stderr)
        return 2
    converter, quantizer, runner = map(Path, sys.argv[1:])
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("SKIP: numpy is required to build the tiny MoE fixture")
        return 77
    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_moe_cuda_"))
    try:
        qwen3_dir = temp_dir / "qwen3_moe"
        qwen3_dir.mkdir()
        build_qwen3_fixture(qwen3_dir)
        qwen35_dir = temp_dir / "qwen35_moe"
        qwen35_dir.mkdir()
        build_qwen35_fixture(qwen35_dir)
        qwen3_package = temp_dir / "tiny_qwen3_moe_w4g32.mollm"
        qwen35_package = temp_dir / "tiny_qwen35_moe_w4g32.mollm"
        qwen3_w8_package = temp_dir / "tiny_qwen3_moe_w8g32.mollm"
        environment = os.environ.copy()
        environment["MOLLM_QUANT_HELPER"] = str(quantizer)
        environment["MOLLM_QUANT_THREADS"] = "1"
        for model_dir, package, quant in (
                (qwen3_dir, qwen3_package, "w4g32"),
                (qwen35_dir, qwen35_package, "w4g32"),
                (qwen3_dir, qwen3_w8_package, "w8g32")):
            subprocess.run(
                [sys.executable, str(converter), str(model_dir), str(package),
                 quant], check=True, env=environment,
                cwd=str(converter.parent.parent))
            if quant == "w4g32":
                assert_has_bg32_weight(package)
            else:
                assert_has_int8_weight(package)
        runner_environment = environment.copy()
        runner_environment["MOLLM_CUDA_PROFILE"] = "1"
        completed = subprocess.run(
            [str(runner), str(qwen3_package), str(qwen35_package),
             str(qwen3_w8_package)],
            check=False, env=runner_environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode == 77:
            return 77
        completed.check_returncode()
        if "  fallback " in completed.stderr:
            raise AssertionError(
                "tiny CUDA MoE E2E used an operator fallback")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
