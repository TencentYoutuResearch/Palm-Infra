#!/usr/bin/env python3
"""Smoke tests for the Qwen3 decoder-only graph builder."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen3 import build_graph
from transpile import OpType


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def tiny_cfg():
    return {
        "model_type": "qwen3",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "vocab_size": 128,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-6,
    }


def main():
    g = build_graph(".", tiny_cfg(), seq_len=8, n_ctx=64, is_prefill=True)

    sdpa_nodes = [n for n in g._nodes if n.op_type == OpType.SDPA]
    check(len(sdpa_nodes) == 2, "one SDPA node per layer")
    for node in sdpa_nodes:
        check(node.params_i32[2] == 4, "SDPA num_heads")
        check(node.params_i32[3] == 2, "SDPA num_kv_heads")
        check(node.params_i32[4] == 4, "SDPA head_dim")

    q_proj = [
        n for n in g._nodes
        if n.op_type == OpType.CONSTANT
        and n.params_str
        and n.params_str[0].endswith("model_layers_0_self_attn_q_proj_weight.weights")
    ][0]
    check(tuple(q_proj.out_shape[:2]) == (16, 16),
          "Qwen3 q_proj is [num_heads*head_dim, hidden], not gated 2x")

    cache_k = [
        n for n in g._nodes
        if n.op_type == OpType.INPUT
        and n.params_str
        and n.params_str[0] == "cache_k0"
    ][0]
    check(tuple(cache_k.out_shape[:3]) == (4, 64, 2),
          "KV cache uses num_key_value_heads")

    check(not any(n.op_type == OpType.SIGMOID for n in g._nodes),
          "Qwen3 attention has no Qwen3.5 output gate")

    print("Qwen3 graph tests passed")


if __name__ == "__main__":
    main()
