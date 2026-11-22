#!/usr/bin/env python3
"""Structural tests for the Qwen3.5 graph builder."""

import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen35 import (
    _query_then_gate_rows, build_graph, build_mtp_graph,
    build_mtp_streamed_weights,
)
from transpile import OpType


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def tiny_cfg(layer_type="linear_attention"):
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "layer_types": [layer_type],
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
            },
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": 32,
            "vocab_size": 128,
        },
    }


def mtp_cfg():
    cfg = tiny_cfg("full_attention")
    cfg["text_config"]["mtp_num_hidden_layers"] = 1
    return cfg


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    payload = b""
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float16).tobytes()
        header[name] = {"dtype": "F16", "shape": list(array.shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def streamed_array(weight, dtype) -> np.ndarray:
    output = io.BytesIO()
    weight.write_to(output)
    return np.frombuffer(output.getvalue()[88:], dtype=dtype).reshape(weight.logical_shape)


def check_mtp_graph():
    graph = build_mtp_graph(".", mtp_cfg(), seq_len=8, n_ctx=64)
    inputs = [node.params_str[0] for node in graph._nodes if node.op_type == OpType.INPUT]
    check(inputs == ["hidden", "mask", "cos", "sin", "cache_k0", "cache_v0", "target_hidden"],
          "Qwen3.5 MTP graph exposes the engine's draft inputs")
    references = {node.weight_path for node in graph._nodes if node.weight_path}
    check(not any("model_language_model" in name for name in references),
          "MTP weights never alias target layer weights")
    check({"./embed_tokens.weights", "./lm_head.weights"} <= references,
          "MTP graph shares the target embedding and head")
    consumed = {i for node in graph._nodes for i in node.inputs}
    outputs = [node for node in graph._nodes
               if node.id not in consumed and node.op_type not in (OpType.INPUT, OpType.CONSTANT)]
    check(outputs[-1].op_type == OpType.ADD_RMS_NORM and outputs[-1].out_shape[0] == 16,
          "MTP graph ends with the mtp.norm state")
    try:
        build_mtp_graph(".", tiny_cfg("full_attention"))
    except ValueError:
        pass
    else:
        raise AssertionError("Qwen3.5 MTP export requires a declared draft layer")


def check_mtp_weights():
    cfg = mtp_cfg()
    tc = cfg["text_config"]
    hidden, heads, head_dim = tc["hidden_size"], tc["num_attention_heads"], tc["head_dim"]
    kv = tc["num_key_value_heads"] * head_dim
    rng = np.random.default_rng(7)
    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float16)
    base = "mtp.layers.0"
    tensors = {
        "mtp.fc.weight": tensor(hidden, 2 * hidden),
        f"{base}.self_attn.q_proj.weight": tensor(2 * heads * head_dim, hidden),
        f"{base}.self_attn.k_proj.weight": tensor(kv, hidden),
        f"{base}.self_attn.v_proj.weight": tensor(kv, hidden),
        f"{base}.self_attn.o_proj.weight": tensor(hidden, heads * head_dim),
        f"{base}.self_attn.q_norm.weight": tensor(head_dim),
        f"{base}.self_attn.k_norm.weight": tensor(head_dim),
        f"{base}.mlp.gate_proj.weight": tensor(tc["intermediate_size"], hidden),
        f"{base}.mlp.up_proj.weight": tensor(tc["intermediate_size"], hidden),
        f"{base}.mlp.down_proj.weight": tensor(hidden, tc["intermediate_size"]),
    }
    for name in ("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden", "mtp.norm",
                 f"{base}.input_layernorm", f"{base}.post_attention_layernorm"):
        tensors[f"{name}.weight"] = tensor(hidden)
    with tempfile.TemporaryDirectory() as directory:
        write_safetensors(Path(directory) / "model.safetensors", tensors)
        weights = build_mtp_streamed_weights(directory, cfg)
        graph = build_mtp_graph(".", cfg)
        dedicated = {node.weight_path for node in graph._nodes if node.weight_path} - {
            "./embed_tokens.weights", "./lm_head.weights"}
        check(set(weights) == dedicated, "MTP weights match the graph references exactly")
        def get(name, dtype=np.float16):
            return streamed_array(weights[f"./{name}.weights"], dtype)
        qkv = np.concatenate([
            _query_then_gate_rows(tensors[f"{base}.self_attn.q_proj.weight"], heads, head_dim),
            tensors[f"{base}.self_attn.k_proj.weight"], tensors[f"{base}.self_attn.v_proj.weight"]])
        check(np.array_equal(get("mtp_layers_0_self_attn_qkv_proj_weight"), qkv),
              "MTP Q/K/V rows follow the target query-then-gate layout")
        fc = tensors["mtp.fc.weight"]
        check(np.array_equal(get("mtp_fc_embedding_weight"), fc[:, :hidden]) and
              np.array_equal(get("mtp_fc_hidden_weight"), fc[:, hidden:]),
              "mtp.fc splits into its embedding and hidden input columns")
        check(np.array_equal(get("mtp_layers_0_mlp_gate_up_proj_weight"), np.concatenate([
                  tensors[f"{base}.mlp.gate_proj.weight"], tensors[f"{base}.mlp.up_proj.weight"]])),
              "MTP MLP stores gate rows before up rows")
        check(np.array_equal(get("mtp_norm_weight", np.float32),
                             tensors["mtp.norm.weight"].astype(np.float32) + 1.0),
              "MTP norms store the effective 1 + gamma scale")


def main():
    check_mtp_graph()
    check_mtp_weights()
    graph = build_graph(
        ".", tiny_cfg(), seq_len=8, n_ctx=64, is_prefill=True
    )

    constants = [
        node.params_str[0]
        for node in graph._nodes
        if node.op_type == OpType.CONSTANT and node.params_str
    ]
    check(
        any(name.endswith("mlp_gate_up_proj_weight.weights")
            for name in constants),
        "Qwen3.5 uses a merged MLP gate/up weight",
    )
    check(
        not any(name.endswith("mlp_gate_proj_weight.weights")
                or name.endswith("mlp_up_proj_weight.weights")
                for name in constants),
        "Qwen3.5 graph has no separate MLP gate/up weights",
    )

    check(
        sum(node.op_type == OpType.SWIGLU for node in graph._nodes) == 1,
        "one fused SWIGLU per layer",
    )
    check(
        not any(node.op_type in (OpType.SILU, OpType.MUL)
                for node in graph._nodes),
        "MLP no longer dispatches standalone SILU/MUL",
    )
    check(
        sum(node.op_type == OpType.SHORTCONV for node in graph._nodes) == 1
        and sum(node.op_type == OpType.GATED_DELTANET_PREFILL
                for node in graph._nodes) == 1,
        "prefill keeps separate ShortConv and GDN recurrence",
    )
    check(
        sum(node.op_type == OpType.ADD_RMS_NORM
            for node in graph._nodes) == 2,
        "each layer fuses both residual add + RMSNorm pairs",
    )
    check(
        not any(node.op_type == OpType.ADD for node in graph._nodes),
        "Qwen3.5 residual stream has no standalone ADD",
    )
    check(
        any(name.endswith("linear_attn_in_proj_weight.weights")
            for name in constants),
        "Qwen3.5 linear attention uses one merged input projection weight",
    )
    check(
        not any(name.endswith("linear_attn_in_proj_qkv_weight.weights")
                or name.endswith("linear_attn_in_proj_a_weight.weights")
                or name.endswith("linear_attn_in_proj_b_weight.weights")
                or name.endswith("linear_attn_in_proj_ab_weight.weights")
                or name.endswith("linear_attn_in_proj_z_weight.weights")
                for name in constants),
        "Qwen3.5 graph has no separate linear-attention input projections",
    )

    full_graph = build_graph(
        ".", tiny_cfg("full_attention"), seq_len=8, n_ctx=64, is_prefill=True
    )
    full_constants = [
        node.params_str[0]
        for node in full_graph._nodes
        if node.op_type == OpType.CONSTANT and node.params_str
    ]
    check(
        any(name.endswith("self_attn_qkv_proj_weight.weights")
            for name in full_constants),
        "Qwen3.5 full attention uses a merged Q/K/V weight",
    )
    check(
        not any(name.endswith("self_attn_q_proj_weight.weights")
                or name.endswith("self_attn_k_proj_weight.weights")
                or name.endswith("self_attn_v_proj_weight.weights")
                for name in full_constants),
        "Qwen3.5 graph has no separate full-attention Q/K/V weights",
    )
    check(
        sum(node.op_type == OpType.SIGMOID_MUL
            for node in full_graph._nodes) == 1,
        "Qwen3.5 full attention fuses sigmoid and multiply",
    )
    check(
        sum(node.op_type == OpType.QK_RMS_NORM_ROPE
            for node in full_graph._nodes) == 1
        and not any(node.op_type == OpType.RMS_NORM_ROPE
                    for node in full_graph._nodes)
        and not any(node.op_type == OpType.ROTARY_EMBED
                    for node in full_graph._nodes),
        "full attention fuses Q and K RMSNorm/materialization/RoPE",
    )
    check(
        not any(node.op_type == OpType.SIGMOID
                for node in full_graph._nodes),
        "Qwen3.5 full attention has no standalone sigmoid",
    )
    check(
        sum(node.op_type == OpType.CONTIGUOUS
            for node in full_graph._nodes) == 0,
        "full attention should not require standalone materialization",
    )

    decode_graph = build_graph(
        ".", tiny_cfg(), seq_len=1, n_ctx=64, is_prefill=False
    )
    check(
        sum(node.op_type == OpType.GATED_DELTANET_CONV_DECODE
            for node in decode_graph._nodes) == 1,
        "decode fuses ShortConv with the GDN recurrence",
    )
    check(
        not any(node.op_type in (
            OpType.SHORTCONV, OpType.GATED_DELTANET_DECODE)
            for node in decode_graph._nodes),
        "decode has no standalone ShortConv or GDN dispatch",
    )

    print("Qwen3.5 graph tests passed")


if __name__ == "__main__":
    main()
