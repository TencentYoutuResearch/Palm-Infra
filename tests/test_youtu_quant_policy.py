#!/usr/bin/env python3
"""Unit tests for Youtu MLA converter quantization policy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from mla import _quant_spec


def check(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected}, got {actual}")


def main():
    check(_quant_spec("none", 2048), None, "none is unquantized")
    check(_quant_spec("w8pc", 2048), ("w8", 2048), "w8pc uses K as group")
    check(_quant_spec("w8g128", 2048), ("w8", 128), "w8g128 parses")
    check(_quant_spec("w4g128", 2048), ("w4", 128), "w4g128 parses")

    check(_quant_spec("w4mixg128", 2048, "lm_head.weight"), ("w8", 128),
          "mixed W4 promotes explicit lm_head")

    kv_b = "model.layers.7.self_attn.kv_b_proj.weight"
    check(_quant_spec("w4mixg128", 512, kv_b), ("w8", 128),
          "mixed W4 promotes MLA KV expansion")

    o_proj = "model.layers.7.self_attn.o_proj.weight"
    check(_quant_spec("w4mixg128", 2048, o_proj), ("w8", 128),
          "mixed W4 promotes attention output")

    down = "model.layers.7.mlp.down_proj.weight"
    check(_quant_spec("w4mixg128", 6144, down), ("w8", 128),
          "mixed W4 promotes FFN down")

    q_a = "model.layers.7.self_attn.q_a_proj.weight"
    check(_quant_spec("w4mixg128", 2048, q_a), ("w4", 128),
          "mixed W4 leaves Q compression at base W4")

    q_b = "model.layers.7.self_attn.q_b_proj.weight"
    check(_quant_spec("w4mixg128", 1536, q_b), ("w4", 128),
          "mixed W4 leaves Q expansion at base W4")

    kv_a = "model.layers.7.self_attn.kv_a_proj_with_mqa.weight"
    check(_quant_spec("w4mixg128", 2048, kv_a), ("w4", 128),
          "mixed W4 leaves KV compression at base W4")

    gate_up = "model.layers.7.mlp.gate_up_proj.weight"
    check(_quant_spec("w4mixg128", 2048, gate_up), ("w4", 128),
          "mixed W4 leaves merged gate/up at base W4")

    print("Youtu quant policy tests passed")


if __name__ == "__main__":
    main()
