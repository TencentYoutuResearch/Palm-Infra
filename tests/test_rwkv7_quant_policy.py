#!/usr/bin/env python3
"""Unit tests for RWKV7 converter mixed-precision policy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from rwkv7 import _quant_spec


def check(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: got {actual}, expected {expected}")


check(_quant_spec("fp16", "blocks_0_ffn_key_weight", 2048), None,
      "FP16 leaves matrices unquantized")
check(_quant_spec("w8", "blocks_0_ffn_key_weight", 2048), ("w8", 2048),
      "W8 uses per-channel groups")
check(_quant_spec("w4", "embed_tokens", 2048), None,
      "embedding remains FP16")
check(_quant_spec("w4", "lm_head", 2048), ("w8", 128),
      "mixed W4 keeps lm_head W8")
check(_quant_spec("w4", "blocks_0_att_w1", 2048), ("w8", 128),
      "mixed W4 keeps low-rank input W8")
check(_quant_spec("w4", "blocks_0_att_g2", 256), ("w8", 128),
      "mixed W4 keeps low-rank output W8")
check(_quant_spec("w4", "blocks_0_att_key_weight", 2048), ("w4", 128),
      "mixed W4 quantizes large attention matrices")
check(_quant_spec("w4", "blocks_0_ffn_value_weight", 8192), ("w4", 128),
      "mixed W4 quantizes FFN down matrix")

print("RWKV7 quant policy tests passed")
