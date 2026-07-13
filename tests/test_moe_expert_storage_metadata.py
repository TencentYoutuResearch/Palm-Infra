#!/usr/bin/env python3
"""Smoke tests for MoE expert storage package metadata."""

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from transpile import GraphBuilder, Precision, _write_weight_file, save_package


def read_package_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        header = f.read(128)
        meta_off, meta_len = struct.unpack_from("<QQ", header, 8)
        f.seek(meta_off)
        return json.loads(f.read(meta_len))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        weights_dir = Path(tmp) / "weights"
        weights_dir.mkdir()

        gate_up_name = "layer0_experts_gate_up.weights"
        down_name = "layer0_experts_down.weights"

        # Logical gate_up: E=2, rows_per_expert=2, K=4.
        gate_up_q4 = np.arange(4 * 2, dtype=np.uint8)
        gate_up_scales = np.arange(4 * 2, dtype=np.float32)
        _write_weight_file(
            str(weights_dir / gate_up_name),
            gate_up_q4,
            scales=gate_up_scales,
            group_size=2,
            num_groups=gate_up_scales.size,
            precision=Precision.INT4,
            logical_shape=(4, 4),
        )

        # Logical down: E=2, rows_per_expert=3, K=4.
        down_q4 = np.arange(6 * 2, dtype=np.uint8)
        down_scales = np.arange(6 * 2, dtype=np.float32)
        _write_weight_file(
            str(weights_dir / down_name),
            down_q4,
            scales=down_scales,
            group_size=2,
            num_groups=down_scales.size,
            precision=Precision.INT4,
            logical_shape=(6, 4),
        )

        g = GraphBuilder()
        x = g.input("x", (4, 1), prec=Precision.FP32)
        gate_up = g.weight("./" + gate_up_name, (4, 4), Precision.INT4)
        down = g.weight("./" + down_name, (6, 4), Precision.INT4)
        g.matmul(x, gate_up)
        g.matmul(x, down)

        package_path = str(Path(tmp) / "test.mollm")
        save_package(
            package_path,
            g,
            g,
            str(weights_dir),
            {
                "num_experts": 2,
                "moe_expert_storage": {
                    "version": 1,
                    "layout": "aggregate_rows_v1",
                    "num_experts": 2,
                    "layers": [{
                        "layer": 0,
                        "num_experts": 2,
                        "gate_up": {
                            "weight": "./" + gate_up_name,
                            "rows_per_expert": 2,
                            "cols": 4,
                        },
                        "down": {
                            "weight": "./" + down_name,
                            "rows_per_expert": 3,
                            "cols": 4,
                        },
                    }],
                },
            },
        )

        metadata = read_package_metadata(package_path)
        layer = metadata["moe_expert_storage"]["layers"][0]
        gate_up_meta = layer["gate_up"]
        down_meta = layer["down"]

        assert gate_up_meta["precision"] == int(Precision.INT4)
        assert gate_up_meta["shape"][:2] == [4, 4]
        assert gate_up_meta["data_offset"] == 88
        assert gate_up_meta["scales_offset"] == 88 + gate_up_q4.nbytes
        assert gate_up_meta["group_size"] == 2
        assert gate_up_meta["groups_per_row"] == 2
        assert gate_up_meta["expert_data_bytes"] == gate_up_q4.nbytes // 2
        assert gate_up_meta["expert_scales_bytes"] == gate_up_scales.nbytes // 2

        assert down_meta["shape"][:2] == [6, 4]
        assert down_meta["expert_data_bytes"] == down_q4.nbytes // 2
        assert down_meta["expert_scales_bytes"] == down_scales.nbytes // 2
        assert down_meta["weight_offset"] > gate_up_meta["weight_offset"]

    print("MoE expert storage metadata tests passed")


if __name__ == "__main__":
    main()
