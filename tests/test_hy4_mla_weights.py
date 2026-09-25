#!/usr/bin/env python3
"""Check the byte layout written for absorbed Hy4 MLA weights."""

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from gguf_reader import torch_to_gguf
from safetensors_stream import SafeTensorSlice, split_mla_kv_b
from transpile import WeightByteRange


class FakeIndex:
    def __init__(self, tensor):
        self.source = tensor

    def tensor(self, name):
        assert name == self.source.name
        return self.source


def test_split(dtype):
    heads, nope, v_dim, rank = 3, 5, 7, 9
    values = (np.arange(heads * (nope + v_dim) * rank, dtype=np.float32)
              .reshape(heads, nope + v_dim, rank) / 128 - 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kv_b.bin"
        if dtype == "BF16":
            raw = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
            decoded = ((np.frombuffer(raw, "<u2").astype(np.uint32) << 16)
                       .view(np.float32).reshape(values.shape))
        else:
            raw = values.astype("<f2").tobytes()
            decoded = np.frombuffer(raw, "<f2").reshape(values.shape)
        path.write_bytes(raw)
        name = "model.layers.0.self_attn.kv_b_proj.weight"
        source = SafeTensorSlice(
            name, dtype, (heads * (nope + v_dim), rank),
            WeightByteRange(str(path), 0, len(raw)))
        k_b, v_b = split_mla_kv_b(
            FakeIndex(source), name, num_heads=heads, nope_dim=nope,
            v_head_dim=v_dim, kv_lora_rank=rank)
        assert k_b.logical_shape == (heads, rank, nope)
        assert v_b.logical_shape == (heads, v_dim, rank)

        output = io.BytesIO()
        k_b._copy_ranges(output, k_b.data_ranges, chunk_size=17)
        actual_k = np.frombuffer(output.getvalue(), "<f2").reshape(k_b.logical_shape)
        np.testing.assert_array_equal(
            actual_k, decoded[:, :nope, :].transpose(0, 2, 1).astype("<f2"))

        output = io.BytesIO()
        v_b._copy_ranges(output, v_b.data_ranges, chunk_size=18)
        actual_v = np.frombuffer(output.getvalue(), "<f2").reshape(v_b.logical_shape)
        np.testing.assert_array_equal(
            actual_v, decoded[:, nope:, :].astype("<f2"))


def main():
    test_split("BF16")
    test_split("F16")
    assert torch_to_gguf("model.layers.7.self_attn.k_b.weight") == (
        "blk.7.attn_k_b.weight")
    assert torch_to_gguf("model.layers.7.self_attn.v_b.weight") == (
        "blk.7.attn_v_b.weight")
    print("Hy4 absorbed MLA weight layout tests passed")


if __name__ == "__main__":
    main()
