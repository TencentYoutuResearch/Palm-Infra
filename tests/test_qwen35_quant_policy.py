#!/usr/bin/env python3
"""Unit tests for Qwen3.5 converter quantization policy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen35 import _quant_spec


def check(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected}, got {actual}")


def main():
    check(_quant_spec("fp16", 1024), None, "fp16 is unquantized")
    check(_quant_spec("none", 1024), None, "none is unquantized")
    check(_quant_spec("w8pc", 1024), ("w8", 1024), "w8pc uses K as group")
    check(_quant_spec("w8g128", 1024), ("w8", 128), "w8g128 parses")
    check(_quant_spec("w4g128", 1024), ("w4", 128), "w4g128 parses")

    check(_quant_spec("w4mixg128", 1024, "lm_head.weight"), ("w8", 128),
          "mixed W4 promotes explicit lm_head")

    qkv = "model.language_model.layers.4.linear_attn.in_proj_qkv.weight"
    check(_quant_spec("w4mixg128", 1024, qkv), ("w8", 128),
          "mixed W4 promotes linear-attention QKV")

    v_proj = "model.language_model.layers.7.self_attn.v_proj.weight"
    check(_quant_spec("w4mixg128", 1024, v_proj), ("w8", 128),
          "mixed W4 promotes full-attention V")

    q_proj = "model.language_model.layers.7.self_attn.q_proj.weight"
    check(_quant_spec("w4mixg128", 1024, q_proj), ("w4", 128),
          "mixed W4 leaves full-attention Q at base W4")

    o_proj = "model.language_model.layers.7.self_attn.o_proj.weight"
    check(_quant_spec("w4mixg128", 1024, o_proj), ("w8", 128),
          "mixed W4 promotes full-attention output")

    linear_out = "model.language_model.layers.4.linear_attn.out_proj.weight"
    check(_quant_spec("w4mixg128", 1024, linear_out), ("w8", 128),
          "mixed W4 promotes linear-attention output")

    early_down = "model.language_model.layers.1.mlp.down_proj.weight"
    middle_down = "model.language_model.layers.4.mlp.down_proj.weight"
    selected_down = "model.language_model.layers.5.mlp.down_proj.weight"
    late_down = "model.language_model.layers.22.mlp.down_proj.weight"
    check(_quant_spec("w4mixg128", 1024, early_down), ("w8", 128),
          "mixed W4 promotes FFN down")
    check(_quant_spec("w4mixg128", 1024, middle_down), ("w8", 128),
          "mixed W4 promotes middle FFN down")
    check(_quant_spec("w4mixg128", 1024, selected_down), ("w8", 128),
          "mixed W4 promotes selected FFN down")
    check(_quant_spec("w4mixg128", 1024, late_down), ("w8", 128),
          "mixed W4 promotes late FFN down")

    print("Qwen3.5 quant policy tests passed")


if __name__ == "__main__":
    main()
