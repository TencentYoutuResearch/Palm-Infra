#!/usr/bin/env python3
"""Build tiny quantized MoE packages and compare CPU/CUDA."""

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


def assert_has_bg128_weight(package: Path) -> None:
    data = package.read_bytes()
    meta_offset, meta_length = struct.unpack_from("<QQ", data, 8)
    weights_offset, _ = struct.unpack_from("<QQ", data, 88)
    metadata = json.loads(data[meta_offset:meta_offset + meta_length])
    for relative_offset, size in metadata["weights"].values():
        if size < 88:
            continue
        flags, _, precision = struct.unpack_from(
            "<III", data, weights_offset + relative_offset + 4)
        if precision == 3 and flags & (1 << 1):
            return
    raise AssertionError("tiny W4G128 package contains no BG128 weight")


def build_qwen3_fixture(model_dir: Path, hidden: int = 32) -> None:
    import numpy as np

    heads = 4
    head_dim = hidden // heads
    kv_heads = 1
    intermediate = 2 * hidden
    moe_intermediate = hidden
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
    """Build a tiny DeepSeek graph that exercises the complete CUDA path."""
    import numpy as np

    sys.path.insert(0, str(models_dir))
    from transpile import (  # pylint: disable=import-outside-toplevel
        DimExpr,
        GraphBuilder,
        Precision,
        WEIGHT_FLAG_FP8_BLOCK128,
        _write_weight_file,
        save_package,
    )

    hidden = 32
    intermediate = 32
    shared_intermediate = 32
    experts = 4
    top_k = 2
    vocab = 8
    hc_mult = 3
    wide = hidden * hc_mult
    hc_mix = (2 + hc_mult) * hc_mult
    dsv_head_dim = hidden
    dsv_ratio = 4
    dsv_projected = 2 * dsv_head_dim
    rng = np.random.default_rng(20260802)

    def matrix(rows: int, cols: int) -> object:
        return rng.uniform(-0.12, 0.12, (rows, cols)).astype(np.float16)

    weights = {
        "embed_tokens.weights": matrix(vocab, hidden),
        "lm_head.weights": matrix(vocab, hidden),
        "layer_0_router.weights": matrix(experts, hidden),
        "dsv_index_weights.weights": np.zeros(
            (1, hidden), dtype=np.float16),
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
        "dsv_sink.weights": np.array([-0.125], dtype=np.float32),
        "dsv_compressor_wkv.weights": rng.uniform(
            -0.08, 0.08,
            (dsv_projected, hidden)).astype(np.float32),
        "dsv_compressor_wgate.weights": rng.uniform(
            -0.04, 0.04,
            (dsv_projected, hidden)).astype(np.float32),
        "dsv_compressor_ape.weights": rng.uniform(
            -0.03, 0.03,
            (dsv_ratio, dsv_projected)).astype(np.float32),
        "dsv_compressor_norm.weights": np.ones(
            dsv_head_dim, dtype=np.float32),
        "dsv_index_wkv.weights": rng.uniform(
            -0.08, 0.08,
            (dsv_projected, hidden)).astype(np.float32),
        "dsv_index_wgate.weights": rng.uniform(
            -0.04, 0.04,
            (dsv_projected, hidden)).astype(np.float32),
        "dsv_index_ape.weights": rng.uniform(
            -0.03, 0.03,
            (dsv_ratio, dsv_projected)).astype(np.float32),
        "dsv_index_norm.weights": np.ones(
            dsv_head_dim, dtype=np.float32),
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

    def write_mxfp4(name: str, rows: int, width: int, seed: int) -> None:
        if width % 32:
            raise AssertionError("MXFP4 fixture width must be divisible by 32")
        local_rng = np.random.default_rng(seed)
        low = local_rng.integers(
            0, 16, size=(rows, width // 2), dtype=np.uint8)
        high = local_rng.integers(
            0, 16, size=(rows, width // 2), dtype=np.uint8)
        packed = low | (high << np.uint8(4))
        scales = local_rng.integers(
            124, 127, size=rows * (width // 32), dtype=np.uint8)
        _write_weight_file(
            str(weights_dir / name), packed, scales=scales,
            group_size=32, num_groups=scales.size,
            precision=Precision.MXFP4, logical_shape=(rows, width),
            scale_dtype=np.uint8)

    def write_fp8(name: str, rows: int, width: int, seed: int) -> None:
        local_rng = np.random.default_rng(seed)
        codes = np.array([
            0x00, 0x20, 0x28, 0x30, 0x38, 0x40,
            0xA0, 0xA8, 0xB0, 0xB8, 0xC0,
        ], dtype=np.uint8)
        values = local_rng.choice(codes, size=(rows, width))
        scale_count = ((rows + 127) // 128) * ((width + 127) // 128)
        scales = local_rng.integers(
            124, 127, size=scale_count, dtype=np.uint8)
        _write_weight_file(
            str(weights_dir / name), values, scales=scales,
            group_size=128, num_groups=scales.size,
            precision=Precision.FP8_E4M3, logical_shape=(rows, width),
            flags=WEIGHT_FLAG_FP8_BLOCK128, scale_dtype=np.uint8)

    write_mxfp4(
        "layer_0_experts_gate_up.weights",
        experts * 2 * intermediate, hidden, 20260803)
    write_mxfp4(
        "layer_0_experts_down.weights",
        experts * hidden, intermediate, 20260804)
    write_fp8(
        "layer_0_shared_experts_gate.weights",
        shared_intermediate, hidden, 20260805)
    write_fp8(
        "layer_0_shared_experts_up.weights",
        shared_intermediate, hidden, 20260806)
    write_fp8(
        "layer_0_shared_experts_down.weights",
        hidden, shared_intermediate, 20260807)
    write_fp8(
        "layer_0_grouped.weights", hidden, hidden // 4, 20260808)
    write_fp8(
        "dsv_index_wq_b.weights", dsv_head_dim, hidden, 20260809)
    for name, values in fp32_weights.items():
        logical_shape = None
        if name.endswith("_ape.weights"):
            logical_shape = (dsv_projected, dsv_ratio)
        _write_weight_file(
            str(weights_dir / name), values, precision=Precision.FP32,
            logical_shape=logical_shape)
    _write_weight_file(
        str(weights_dir / "layer_0_token_hash.weights"), hash_values,
        precision=Precision.INT32, logical_shape=(top_k, vocab))

    def graph(seq_len: int, dynamic: bool) -> object:
        g = GraphBuilder()
        g.set_model_config(
            hidden_size=hidden, num_layers=1, vocab_size=vocab,
            model_type="deepseek_v4", rope_dim=0, rope_theta=10000.0)
        g.weight("./embed_tokens.weights", (vocab, hidden), Precision.FP16)
        g.weight("./lm_head.weights", (vocab, hidden), Precision.FP16)
        hidden_input = g.input(
            "hidden", (hidden, seq_len), Precision.FP32,
            dynamic=(DimExpr.const(), DimExpr.seq()) if dynamic else None)
        token_ids = g.input(
            "token_ids", (seq_len,), Precision.INT32,
            dynamic=(DimExpr.seq(),) if dynamic else None)
        position = g.input("position", (1,), Precision.INT32)
        n_tokens = g.input("n_tokens", (1,), Precision.INT32)

        window_cache = g.input(
            "aux_state0", (dsv_head_dim, 3), Precision.FP32)
        compressor_kv_state = g.input(
            "aux_state1", (dsv_projected, 2 * dsv_ratio),
            Precision.FP32)
        compressor_score_state = g.input(
            "aux_state2", (dsv_projected, 2 * dsv_ratio),
            Precision.FP32)
        compressed_cache = g.input(
            "aux_state3", (dsv_head_dim, 2), Precision.FP32)
        index_kv_state = g.input(
            "aux_state4", (dsv_projected, 2 * dsv_ratio),
            Precision.FP32)
        index_score_state = g.input(
            "aux_state5", (dsv_projected, 2 * dsv_ratio),
            Precision.FP32)
        index_cache = g.input(
            "aux_state6", (dsv_head_dim, 2), Precision.FP32)
        compressor_wkv = g.weight(
            "./dsv_compressor_wkv.weights",
            (dsv_projected, hidden), Precision.FP32)
        compressor_wgate = g.weight(
            "./dsv_compressor_wgate.weights",
            (dsv_projected, hidden), Precision.FP32)
        compressor_ape = g.weight(
            "./dsv_compressor_ape.weights",
            (dsv_projected, dsv_ratio), Precision.FP32)
        compressor_norm = g.weight(
            "./dsv_compressor_norm.weights",
            (dsv_head_dim,), Precision.FP32)
        compressor_dependency = g.dsv4_compressor(
            hidden_input, compressor_wkv, compressor_wgate,
            compressor_ape, compressor_norm, compressor_kv_state,
            compressor_score_state, compressed_cache, position, n_tokens,
            hidden, dsv_head_dim, dsv_ratio, True, False, 0, 0,
            1e-6, 10000.0, 1.0, 32.0, 1.0)
        index_wq_b = g.weight(
            "./dsv_index_wq_b.weights",
            (dsv_head_dim, hidden), Precision.FP8_E4M3)
        index_weights = g.weight(
            "./dsv_index_weights.weights", (1, hidden), Precision.FP16)
        index_wkv = g.weight(
            "./dsv_index_wkv.weights",
            (dsv_projected, hidden), Precision.FP32)
        index_wgate = g.weight(
            "./dsv_index_wgate.weights",
            (dsv_projected, hidden), Precision.FP32)
        index_ape = g.weight(
            "./dsv_index_ape.weights",
            (dsv_projected, dsv_ratio), Precision.FP32)
        index_norm = g.weight(
            "./dsv_index_norm.weights", (dsv_head_dim,), Precision.FP32)
        compressed_indices = g.dsv4_indexer(
            hidden_input, hidden_input, index_wq_b, index_weights,
            index_wkv, index_wgate, index_ape, index_norm,
            index_kv_state, index_score_state, index_cache,
            position, n_tokens, hidden, hidden, 1, dsv_head_dim, top_k,
            dsv_ratio, True, True, 0, 0,
            1e-6, 10000.0, 1.0, 32.0, 1.0)
        attended = g.dsv4_sparse_attention(
            hidden_input, hidden_input, g.weight(
                "./dsv_sink.weights", (1,), Precision.FP32),
            window_cache, position, n_tokens, 1, dsv_head_dim, 3,
            dsv_ratio, top_k, 0, 0, dsv_head_dim ** -0.5, 1e-6,
            10000.0, 1.0, 32.0, 1.0,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            dependencies=[compressor_dependency])
        grouped_weight = g.weight(
            "./layer_0_grouped.weights", (hidden, hidden // 4),
            Precision.FP8_E4M3)
        projected_hidden = g.dsv4_grouped_linear(
            attended, grouped_weight, 4)
        hc_fn = g.weight(
            "./hc_fn.weights", (hc_mix, wide), Precision.FP32)
        hc_scale = g.weight(
            "./hc_scale.weights", (3,), Precision.FP32)
        hc_base = g.weight(
            "./hc_base.weights", (hc_mix,), Precision.FP32)
        residual = g.tile(projected_hidden, (hc_mult, 1))
        packed = g.hc_pre(
            residual, hc_fn, hc_scale, hc_base, hidden, hc_mult,
            sinkhorn_iters=5, norm_eps=1e-6, sinkhorn_eps=1e-6)
        reduced = g.slice_range(packed, 0, hidden, dim=0)
        router = g.weight(
            "./layer_0_router.weights", (experts, hidden), Precision.FP16)
        gate_up = g.weight(
            "./layer_0_experts_gate_up.weights",
            (experts * 2 * intermediate, hidden), Precision.MXFP4)
        down = g.weight(
            "./layer_0_experts_down.weights",
            (experts * hidden, intermediate), Precision.MXFP4)
        shared_gate = g.weight(
            "./layer_0_shared_experts_gate.weights",
            (shared_intermediate, hidden), Precision.FP8_E4M3)
        shared_up = g.weight(
            "./layer_0_shared_experts_up.weights",
            (shared_intermediate, hidden), Precision.FP8_E4M3)
        shared_down = g.weight(
            "./layer_0_shared_experts_down.weights",
            (hidden, shared_intermediate), Precision.FP8_E4M3)
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
            "architecture": "deepseek-v4",
            "num_layers": 1,
            "hidden_size": hidden,
            "num_heads": 1,
            "head_dim": dsv_head_dim,
            "n_ctx": 8,
            "vocab_size": vocab,
            "num_experts": experts,
            "prefill_seq_len": 4,
            "quantization": "native-fp8-mxfp4",
            "moe_expert_storage": {
                "version": 1,
                "layout": "aggregate_rows_v1",
                "num_experts": experts,
                "layers": [{
                    "layer": 0,
                    "num_experts": experts,
                    "gate_up": {
                        "weight": "./layer_0_experts_gate_up.weights",
                        "rows_per_expert": 2 * intermediate,
                        "cols": hidden,
                    },
                    "down": {
                        "weight": "./layer_0_experts_down.weights",
                        "rows_per_expert": hidden,
                        "cols": intermediate,
                    },
                }],
            },
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
        qwen3_g128_dir = temp_dir / "qwen3_moe_g128"
        qwen3_g128_dir.mkdir()
        build_qwen3_fixture(qwen3_g128_dir, hidden=128)
        qwen35_dir = temp_dir / "qwen35_moe"
        qwen35_dir.mkdir()
        build_qwen35_fixture(qwen35_dir)
        qwen3_package = temp_dir / "tiny_qwen3_moe_w4g32.mollm"
        qwen35_package = temp_dir / "tiny_qwen35_moe_w4g32.mollm"
        qwen3_w8_package = temp_dir / "tiny_qwen3_moe_w8g32.mollm"
        qwen3_w4g128_package = temp_dir / "tiny_qwen3_moe_w4g128.mollm"
        hash_weights_dir = temp_dir / "hash_weights"
        hash_weights_dir.mkdir()
        hash_package = temp_dir / "tiny_hash_moe_native.mollm"
        build_hash_package(hash_weights_dir, hash_package, converter.parent)
        environment = os.environ.copy()
        environment["MOLLM_QUANT_HELPER"] = str(quantizer)
        environment["MOLLM_QUANT_THREADS"] = "1"
        for model_dir, package, quant in (
                (qwen3_dir, qwen3_package, "w4g32"),
                (qwen35_dir, qwen35_package, "w4g32"),
                (qwen3_dir, qwen3_w8_package, "w8g32"),
                (qwen3_g128_dir, qwen3_w4g128_package, "w4g128")):
            subprocess.run(
                [sys.executable, str(converter), str(model_dir), str(package),
                 quant], check=True, env=environment,
                cwd=str(converter.parent.parent))
            if quant == "w4g32":
                assert_has_bg32_weight(package)
            elif quant == "w4g128":
                assert_has_bg128_weight(package)
            else:
                assert_has_int8_weight(package)
        runner_environment = environment.copy()
        runner_environment["MOLLM_CUDA_PROFILE"] = "1"
        completed = subprocess.run(
            [str(runner), str(qwen3_package), str(qwen35_package),
             str(qwen3_w8_package), str(qwen3_w4g128_package),
             str(hash_package)],
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
