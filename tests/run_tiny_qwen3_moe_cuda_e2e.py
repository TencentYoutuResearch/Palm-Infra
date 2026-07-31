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


def build_hash_package(weights_dir: Path, package: Path,
                       models_dir: Path) -> None:
    """Build a minimal package that exercises engine token/hash plumbing."""
    import numpy as np

    sys.path.insert(0, str(models_dir))
    from transpile import (  # pylint: disable=import-outside-toplevel
        DimExpr,
        GraphBuilder,
        Precision,
        _write_weight_file,
        save_package,
    )

    hidden = 8
    intermediate = 5
    shared_intermediate = 6
    experts = 4
    top_k = 2
    vocab = 8
    hc_mult = 3
    wide = hidden * hc_mult
    hc_mix = (2 + hc_mult) * hc_mult
    rng = np.random.default_rng(20260802)

    def matrix(rows: int, cols: int) -> object:
        return rng.uniform(-0.12, 0.12, (rows, cols)).astype(np.float16)

    weights = {
        "embed_tokens.weights": matrix(vocab, hidden),
        "lm_head.weights": matrix(vocab, hidden),
        "layer_0_router.weights": matrix(experts, hidden),
        "layer_0_experts_gate_up.weights": matrix(
            experts * 2 * intermediate, hidden),
        "layer_0_experts_down.weights": matrix(
            experts * hidden, intermediate),
        "layer_0_shared_experts_gate.weights": matrix(
            shared_intermediate, hidden),
        "layer_0_shared_experts_up.weights": matrix(
            shared_intermediate, hidden),
        "layer_0_shared_experts_down.weights": matrix(
            hidden, shared_intermediate),
    }
    fp32_weights = {
        "hc_fn.weights": rng.uniform(
            -0.1, 0.1, (hc_mix, wide)).astype(np.float32),
        "hc_scale.weights": np.array(
            [0.75, 1.25, -0.55], dtype=np.float32),
        "hc_base.weights": rng.uniform(
            -0.1, 0.1, hc_mix).astype(np.float32),
        "hc_head_fn.weights": rng.uniform(
            -0.1, 0.1, (hc_mult, wide)).astype(np.float32),
        "hc_head_scale.weights": np.array([0.9], dtype=np.float32),
        "hc_head_base.weights": rng.uniform(
            -0.1, 0.1, hc_mult).astype(np.float32),
    }
    # Physical source layout is [vocab, top_k], while mollm tensor shape is
    # [top_k, vocab] because dimension zero is the contiguous dimension.
    hash_values = np.array([
        [3, 1], [2, 0], [1, 3], [3, 0],
        [0, 2], [2, 1], [1, 0], [0, 3],
    ], dtype=np.int32)
    for name, values in weights.items():
        _write_weight_file(
            str(weights_dir / name), values, precision=Precision.FP16)
    for name, values in fp32_weights.items():
        _write_weight_file(
            str(weights_dir / name), values, precision=Precision.FP32)
    _write_weight_file(
        str(weights_dir / "layer_0_token_hash.weights"), hash_values,
        precision=Precision.INT32, logical_shape=(top_k, vocab))

    def graph(seq_len: int, dynamic: bool) -> object:
        g = GraphBuilder()
        g.set_model_config(
            hidden_size=hidden, num_layers=1, vocab_size=vocab,
            model_type="qwen3_moe", rope_dim=0, rope_theta=10000.0)
        g.weight("./embed_tokens.weights", (vocab, hidden), Precision.FP16)
        g.weight("./lm_head.weights", (vocab, hidden), Precision.FP16)
        hidden_input = g.input(
            "hidden", (hidden, seq_len), Precision.FP32,
            dynamic=(DimExpr.const(), DimExpr.seq()) if dynamic else None)
        token_ids = g.input(
            "token_ids", (seq_len,), Precision.INT32,
            dynamic=(DimExpr.seq(),) if dynamic else None)
        hc_fn = g.weight(
            "./hc_fn.weights", (hc_mix, wide), Precision.FP32)
        hc_scale = g.weight(
            "./hc_scale.weights", (3,), Precision.FP32)
        hc_base = g.weight(
            "./hc_base.weights", (hc_mix,), Precision.FP32)
        residual = g.tile(hidden_input, (hc_mult, 1))
        packed = g.hc_pre(
            residual, hc_fn, hc_scale, hc_base, hidden, hc_mult,
            sinkhorn_iters=5, norm_eps=1e-6, sinkhorn_eps=1e-6)
        reduced = g.slice_range(packed, 0, hidden, dim=0)
        router = g.weight(
            "./layer_0_router.weights", (experts, hidden), Precision.FP16)
        gate_up = g.weight(
            "./layer_0_experts_gate_up.weights",
            (experts * 2 * intermediate, hidden), Precision.FP16)
        down = g.weight(
            "./layer_0_experts_down.weights",
            (experts * hidden, intermediate), Precision.FP16)
        shared_gate = g.weight(
            "./layer_0_shared_experts_gate.weights",
            (shared_intermediate, hidden), Precision.FP16)
        shared_up = g.weight(
            "./layer_0_shared_experts_up.weights",
            (shared_intermediate, hidden), Precision.FP16)
        shared_down = g.weight(
            "./layer_0_shared_experts_down.weights",
            (hidden, shared_intermediate), Precision.FP16)
        hash_table = g.weight(
            "./layer_0_token_hash.weights", (top_k, vocab),
            Precision.INT32)
        branch = g.moe(
            reduced, router, gate_up, down,
            shared_gate, shared_up, shared_down, None,
            hidden_size=hidden, num_experts=experts, top_k=top_k,
            intermediate_size=intermediate,
            shared_intermediate_size=shared_intermediate,
            router_score_func=2, norm_topk_prob=True,
            has_shared_expert=True, shared_expert_has_gate=False,
            routed_scaling_factor=0.7, swiglu_limit=0.5,
            hash_token_ids=token_ids, hash_table=hash_table)
        combined = g.hc_post(branch, residual, packed, hidden, hc_mult)
        hc_head_fn = g.weight(
            "./hc_head_fn.weights", (hc_mult, wide), Precision.FP32)
        hc_head_scale = g.weight(
            "./hc_head_scale.weights", (1,), Precision.FP32)
        hc_head_base = g.weight(
            "./hc_head_base.weights", (hc_mult,), Precision.FP32)
        g.hc_head(
            combined, hc_head_fn, hc_head_scale, hc_head_base,
            hidden, hc_mult, norm_eps=1e-6, hc_eps=1e-6)
        return g

    save_package(
        str(package), graph(4, True), graph(1, False), str(weights_dir),
        {
            "model_name": "tiny-hash-moe",
            "architecture": "qwen3-moe",
            "num_layers": 1,
            "hidden_size": hidden,
            "vocab_size": vocab,
            "prefill_seq_len": 4,
            "quantization": "fp16",
        })


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
        hash_weights_dir = temp_dir / "hash_weights"
        hash_weights_dir.mkdir()
        hash_package = temp_dir / "tiny_hash_moe_fp16.mollm"
        build_hash_package(hash_weights_dir, hash_package, converter.parent)
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
             str(qwen3_w8_package), str(hash_package)],
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
