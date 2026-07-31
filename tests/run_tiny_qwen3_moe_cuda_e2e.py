#!/usr/bin/env python3
"""Build a tiny W4G32 Qwen3-MoE package and compare CPU/CUDA inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from run_tiny_qwen3_cpu_e2e import assert_has_bg32_weight, write_safetensors


def build_fixture(model_dir: Path) -> None:
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
    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_qwen3_moe_cuda_"))
    try:
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        build_fixture(model_dir)
        package = temp_dir / "tiny_qwen3_moe_w4g32.mollm"
        environment = os.environ.copy()
        environment["MOLLM_QUANT_HELPER"] = str(quantizer)
        environment["MOLLM_QUANT_THREADS"] = "1"
        subprocess.run(
            [sys.executable, str(converter), str(model_dir), str(package),
             "w4g32"], check=True, env=environment,
            cwd=str(converter.parent.parent))
        assert_has_bg32_weight(package)
        completed = subprocess.run([str(runner), str(package)], check=False)
        if completed.returncode == 77:
            return 77
        completed.check_returncode()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
