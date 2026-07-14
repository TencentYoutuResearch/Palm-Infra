#!/usr/bin/env python3
"""Smoke tests for the Qwen3-MoE graph builder."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen3_moe import ROUTER_SCORE_SIGMOID, ROUTER_SCORE_SOFTMAX, build_graph
from transpile import OpType


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def tiny_cfg():
    return {
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "moe_intermediate_size": 6,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "n_shared_experts": 0,
        "_has_router_correction_bias": True,
        "_attn_norm_kind": "q_layernorm",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sigmoid",
        "vocab_size": 128,
        "rope_theta": 10000,
        "rope_interleave": False,
        "rms_norm_eps": 1e-6,
    }


def main():
    g = build_graph(".", tiny_cfg(), seq_len=8, n_ctx=64, is_prefill=True)

    moe_nodes = [n for n in g._nodes if n.op_type == OpType.MOE]
    check(len(moe_nodes) == 2, "layers after first_k_dense_replace use MOE")
    for node in moe_nodes:
        check(node.params_i32[1] == 8, "MOE num_experts")
        check(node.params_i32[2] == 2, "MOE top_k")
        check(node.params_i32[5] == ROUTER_SCORE_SIGMOID, "MOE sigmoid router")
        check(node.params_i32[7] == 0, "MOE has no shared expert")
        check(len(node.inputs) == 9, "MOE includes optional router bias")

    sdpa_nodes = [n for n in g._nodes if n.op_type == OpType.SDPA]
    check(len(sdpa_nodes) == 3, "all layers use full attention")
    for node in sdpa_nodes:
        check(node.params_i32[2] == 4, "SDPA num_heads")
        check(node.params_i32[3] == 2, "SDPA num_kv_heads")

    cfg = tiny_cfg()
    cfg.pop("n_routed_experts")
    cfg.pop("first_k_dense_replace")
    cfg.pop("scoring_func")
    cfg["_has_router_correction_bias"] = False
    cfg["_attn_norm_kind"] = "q_norm"
    cfg["num_experts"] = 8
    g = build_graph(".", cfg, seq_len=8, n_ctx=64, is_prefill=True)
    moe_nodes = [n for n in g._nodes if n.op_type == OpType.MOE]
    check(len(moe_nodes) == 3, "official Qwen3-MoE uses MOE from layer 0")
    for node in moe_nodes:
        check(node.params_i32[5] == ROUTER_SCORE_SOFTMAX,
              "official Qwen3-MoE uses softmax router")
        check(len(node.inputs) == 8, "official Qwen3-MoE has no router bias")

    print("Qwen3-MoE graph tests passed")


if __name__ == "__main__":
    main()
