#!/usr/bin/env python3
"""Structural tests for the Qwen3.5 graph builder."""

import os
import struct
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen35 import (
    _QWEN35_MTP_REQUIRED_WEIGHTS,
    _validate_mtp_weights,
    build_graph,
    build_mtp_graph,
)
from transpile import OpType, Precision


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def tiny_cfg(layer_type="linear_attention", mtp_layers=0):
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
            "mtp_num_hidden_layers": mtp_layers,
        },
    }

def official_4b_cfg():
    layer_block = [
        "linear_attention", "linear_attention", "linear_attention",
        "full_attention",
    ]
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "layer_types": layer_block * 8,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
        },
    }


def find_named_node(graph, op_type, suffix):
    matches = [
        node for node in graph._nodes
        if node.op_type == op_type and node.params_str
        and node.params_str[0].endswith(suffix)
    ]
    check(len(matches) == 1, f"expected one graph node ending in {suffix}")
    return matches[0]


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
    for state_graph, label in ((graph, "prefill"), (decode_graph, "decode")):
        conv_state = next(
            node for node in state_graph._nodes
            if node.op_type == OpType.INPUT and node.params_str == ["gdn_conv0"])
        check(
            conv_state.out_prec == Precision.FP32,
            f"{label} GDN convolution state is serialized as FP32",
        )

    mtp_cfg = tiny_cfg("linear_attention", mtp_layers=1)
    mtp_graph = build_mtp_graph(".", mtp_cfg, seq_len=8, n_ctx=64)
    mtp_inputs = {
        node.params_str[0]
        for node in mtp_graph._nodes
        if node.op_type == OpType.INPUT
    }
    check(
        mtp_inputs == {
            "target_hidden", "hidden", "mask", "cos", "sin",
            "cache_k0", "cache_v0",
        },
        "Qwen3.5 MTP graph exposes target hidden and its private KV cache",
    )
    check(
        mtp_graph._nodes[-1].out_shape[:2] == (16, 8),
        "Qwen3.5 MTP graph returns one normalized hidden per token",
    )
    check(
        sum(node.op_type == OpType.SDPA for node in mtp_graph._nodes) == 1,
        "Qwen3.5 MTP uses one full-attention decoder layer",
    )

    target_with_mtp = build_graph(
        ".", mtp_cfg, seq_len=8, n_ctx=64, is_prefill=True,
        expose_mtp_hidden=True,
    )
    raw_hidden_id = int(target_with_mtp.metadata["mtp_hidden_output_id"])
    check(
        target_with_mtp._nodes[raw_hidden_id].out_shape[:2] == (16, 8),
        "MTP-enabled target graph preserves the pre-final-norm hidden state",
    )

    verify_graph = build_graph(
        ".", mtp_cfg, seq_len=2, n_ctx=64, is_prefill=True,
        expose_mtp_hidden=True, verification=True,
    )
    verify_inputs = {
        node.params_str[0]
        for node in verify_graph._nodes
        if node.op_type == OpType.INPUT
    }
    check(
        {"gdn_state0", "gdn_conv0", "gdn_checkpoint0",
         "gdn_conv_checkpoint0"}.issubset(verify_inputs),
        "verification graph exposes live and checkpoint GDN state",
    )
    verify_nodes = [
        node for node in verify_graph._nodes
        if node.op_type == OpType.GATED_DELTANET_CONV_VERIFY
    ]
    check(len(verify_nodes) == 1, "verification graph serializes one op 113")
    verify_node = verify_nodes[0]
    check(
        int(verify_node.op_type) == 113 and verify_node.params_i32[8] == 1,
        "verification op ABI stores confirmed_prefix_tokens in i32[8]",
    )
    check(
        verify_node.out_shape[:2] == (8, 2),
        "verification op emits both target positions",
    )
    with tempfile.TemporaryDirectory(prefix="qwen35_verify_graph_") as tmp:
        path = os.path.join(tmp, "verify")
        verify_graph.save(path)
        with open(path + ".graph", "rb") as graph_file:
            raw = graph_file.read()
        check(
            struct.pack("<II", verify_node.id, 113) in raw,
            "serialized graph blob contains op type 113",
        )

    official_cfg = official_4b_cfg()
    official_graph = build_graph(
        ".", official_cfg, seq_len=2, n_ctx=64, is_prefill=True
    )
    official_gdn = [
        node for node in official_graph._nodes
        if node.op_type == OpType.GATED_DELTANET_PREFILL
    ]
    official_sdpa = [
        node for node in official_graph._nodes
        if node.op_type == OpType.SDPA
    ]
    check(
        len(official_gdn) == 24 and len(official_sdpa) == 8
        and sum(node.op_type == OpType.SWIGLU
                for node in official_graph._nodes) == 32,
        "official 4B shape builds 32 layers: 24 GDN and 8 SDPA",
    )
    check(
        all(node.params_i32[0] == 16 and node.params_i32[7] == 32
            for node in official_gdn),
        "official 4B GDN uses 16 key heads and 32 value heads",
    )
    check(
        all(node.out_shape[:2] == (4096, 2) for node in official_gdn),
        "official 4B GDN output width is 4096",
    )

    official_state = find_named_node(
        official_graph, OpType.INPUT, "gdn_state0")
    official_conv_state = find_named_node(
        official_graph, OpType.INPUT, "gdn_conv0")
    check(
        official_state.out_shape[:3] == (128, 128, 32),
        "official 4B GDN state shape is [128,128,32]",
    )
    check(
        official_conv_state.out_shape[:2] == (8192, 3),
        "official 4B convolution state shape is [8192,3]",
    )

    official_linear_in = find_named_node(
        official_graph, OpType.CONSTANT,
        "model_language_model_layers_0_linear_attn_in_proj_weight.weights")
    official_gdn_out = find_named_node(
        official_graph, OpType.CONSTANT,
        "model_language_model_layers_0_linear_attn_out_proj_weight.weights")
    official_qkv = find_named_node(
        official_graph, OpType.CONSTANT,
        "model_language_model_layers_3_self_attn_qkv_proj_weight.weights")
    official_mlp = find_named_node(
        official_graph, OpType.CONSTANT,
        "model_language_model_layers_0_mlp_gate_up_proj_weight.weights")
    check(
        official_linear_in.out_shape[:2] == (12352, 2560),
        "official 4B fused GDN input projection is [12352,2560]",
    )
    check(
        official_gdn_out.out_shape[:2] == (2560, 4096),
        "official 4B GDN output projection consumes width 4096",
    )
    check(
        official_qkv.out_shape[:2] == (10240, 2560),
        "official 4B full-attention fused QKV is [10240,2560]",
    )
    check(
        official_mlp.out_shape[:2] == (18432, 2560),
        "official 4B fused MLP gate/up is [18432,2560]",
    )

    official_mtp_graph = build_mtp_graph(
        ".", official_cfg, seq_len=2, n_ctx=64)
    official_mtp_fc = find_named_node(
        official_mtp_graph, OpType.CONSTANT, "mtp_fc_weight.weights")
    official_mtp_qkv = find_named_node(
        official_mtp_graph, OpType.CONSTANT,
        "mtp_layers_0_self_attn_qkv_proj_weight.weights")
    check(
        official_mtp_fc.out_shape[:2] == (2560, 5120),
        "official 4B MTP FC is [2560,5120]",
    )
    check(
        official_mtp_qkv.out_shape[:2] == (10240, 2560),
        "official 4B MTP fused QKV is [10240,2560]",
    )

    complete_names = {name: object() for name in _QWEN35_MTP_REQUIRED_WEIGHTS}
    check(
        _validate_mtp_weights(complete_names, mtp_cfg) == 1,
        "complete Qwen3.5 MTP tensor set is accepted",
    )
    incomplete_names = dict(complete_names)
    incomplete_names.pop("mtp.fc.weight")
    try:
        _validate_mtp_weights(incomplete_names, mtp_cfg)
    except ValueError as error:
        check(
            "mtp.fc.weight" in str(error),
            "missing Qwen3.5 MTP tensor is named in the converter error",
        )
    else:
        raise AssertionError("missing Qwen3.5 MTP tensor was accepted")

    print("Qwen3.5 graph tests passed")


if __name__ == "__main__":
    main()
