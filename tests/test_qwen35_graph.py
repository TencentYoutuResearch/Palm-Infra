#!/usr/bin/env python3
"""Structural tests for the Qwen3.5 graph builder."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen35 import build_graph
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


def main():
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
