#!/usr/bin/env python3
"""Verify streamed NVFP4 expert pair packing and XMAP flags."""

import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from safetensors_stream import (
    SafeTensorSlice, aggregate_nvfp4_experts, aggregate_nvfp4_input_scales,
)
from transpile import (
    WEIGHT_FLAG_EXPERT_INTERLEAVED,
    WEIGHT_FLAG_NVFP4_Q8_PAIR,
    WEIGHT_HEADER_STRUCT,
    WeightByteRange,
    _augment_moe_expert_storage,
    GraphBuilder, Precision, save_package,
)


class FakeIndex:
    def __init__(self, tensors):
        self.tensors = tensors

    def tensor(self, name):
        return self.tensors[name]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "nvfp4.bin"
        payload = bytearray()
        tensors = {}

        def add(name, dtype, shape, data):
            offset = len(payload)
            payload.extend(data)
            tensors[name] = SafeTensorSlice(
                name, dtype, shape,
                WeightByteRange(str(source_path), offset, len(data)))

        original = []
        block_scales = []
        globals_ = []
        for expert in range(2):
            values = (
                np.arange(4 * 16, dtype=np.uint8) + expert * 64
            ).reshape(4, 16)
            scales = np.arange(4 * 2, dtype=np.uint8) + 0x30 + expert * 8
            global_scale = struct.pack("<f", 1.0 + expert)
            name = f"expert.{expert}.weight"
            add(name, "U8", (4, 16), values.tobytes())
            add(name + "_scale", "F8_E4M3", (4, 2), scales.tobytes())
            add(name + "_scale_2", "F32", (), global_scale)
            add(f"expert.{expert}.input_scale", "F32", (),
                struct.pack("<f", .001 * (expert + 1)))
            add(f"expert.{expert}.up.input_scale", "F32", (1,),
                struct.pack("<f", .003 * (expert + 1)))
            original.append(values)
            block_scales.append(scales.tobytes())
            globals_.append(global_scale)

        source_path.write_bytes(payload)
        calibration_names = [[f"expert.{expert}.weight", f"expert.{expert}.up.weight"]
            for expert in range(2)]
        calibration = aggregate_nvfp4_input_scales(FakeIndex(tensors), calibration_names)
        assert calibration.logical_shape == (2, 2)
        calibration_output = io.BytesIO()
        calibration.write_to(calibration_output)
        assert calibration_output.getvalue()[88:] == b"".join(
            struct.pack("<ff", .001 * (expert + 1), .003 * (expert + 1))
            for expert in range(2))
        for invalid in ([], [[]], [["bad_name"]],
                        [["expert.0.weight"], ["expert.1.weight", "expert.1.up.weight"]]):
            try:
                aggregate_nvfp4_input_scales(FakeIndex(tensors), invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid calibration shape/name accepted")
        scale_slice = tensors["expert.0.input_scale"]
        for invalid_value in (0., -1., float("inf"), float("nan")):
            invalid_path = Path(tmp) / "invalid-scale.bin"
            invalid_path.write_bytes(struct.pack("<f", invalid_value))
            invalid_tensors = dict(tensors)
            invalid_tensors[scale_slice.name] = SafeTensorSlice(
                scale_slice.name, "F32", (), WeightByteRange(str(invalid_path), 0, 4))
            try:
                aggregate_nvfp4_input_scales(FakeIndex(invalid_tensors), [["expert.0.weight"]])
            except ValueError:
                pass
            else:
                raise AssertionError("invalid calibration value accepted")
        streamed = aggregate_nvfp4_experts(
            FakeIndex(tensors),
            [[f"expert.{expert}.weight"] for expert in range(2)])
        header = streamed.header()
        assert header["flags"] & WEIGHT_FLAG_EXPERT_INTERLEAVED
        assert header["flags"] & WEIGHT_FLAG_NVFP4_Q8_PAIR
        metadata = {"moe_expert_storage": {"num_experts": 2, "layers": [{
            "gate_up": {"weight": "./experts.weights", "rows_per_expert": 4,
                        "input_scales_weight": "./input.weights"}}]}}
        _augment_moe_expert_storage(metadata,
            {"./experts.weights": [0, streamed.size],
             "./input.weights": [streamed.size, calibration.size]}, {},
            {"./experts.weights": header, "./input.weights": calibration.header()})
        input_metadata = metadata["moe_expert_storage"]["layers"][0]["gate_up"]["input_scales"]
        assert input_metadata["scales_per_expert"] == 2
        assert input_metadata["data_bytes"] == 16
        assert input_metadata["weight_offset"] == streamed.size
        assert input_metadata["data_offset"] == 88
        # Exercise the actual package writer: an unused metadata constant must
        # still appear in the offset map and retain every calibrated byte.
        graph = GraphBuilder()
        graph.input("hidden", (32, 1))
        graph.weight("./experts.weights", streamed.logical_shape, Precision.NVFP4)
        graph.weight("./input.weights", calibration.logical_shape, Precision.FP32)
        package_path = Path(tmp) / "calibrated.mollm"
        save_package(str(package_path), graph, graph, tmp, metadata,
            streamed_weights={"./experts.weights": streamed, "./input.weights": calibration})
        package_bytes = package_path.read_bytes()
        package_header = struct.unpack("<II15Q", package_bytes[:128])
        metadata_offset, metadata_size = package_header[2:4]
        package_metadata = json.loads(package_bytes[metadata_offset:metadata_offset + metadata_size])
        packed_input = package_metadata["moe_expert_storage"]["layers"][0]["gate_up"]["input_scales"]
        scale_offset = package_header[12] + packed_input["weight_offset"] + packed_input["data_offset"]
        assert package_bytes[scale_offset:scale_offset + packed_input["data_bytes"]] == calibration_output.getvalue()[88:]

        output = io.BytesIO()
        streamed.write_to(output)
        raw = output.getvalue()
        unpacked = WEIGHT_HEADER_STRUCT.unpack(raw[:88])
        assert unpacked[1] == header["flags"]

        expected = bytearray()
        for expert in range(2):
            values = original[expert]
            pair_packed = np.ascontiguousarray(
                values.reshape(4, 2, 8).transpose(1, 0, 2)
            ).tobytes()
            expected.extend(pair_packed)
            expected.extend(block_scales[expert])
            expected.extend(globals_[expert] * 4)
        assert raw[88:] == expected

        # Exercise aligned streaming with a deliberately non-aligned generic
        # chunk size; every transform call must still contain whole row tiles.
        aligned = io.BytesIO()
        streamed._copy_ranges(aligned, streamed.data_ranges, chunk_size=37)
        expected_data = bytearray()
        for values in original:
            expected_data.extend(np.ascontiguousarray(
                values.reshape(4, 2, 8).transpose(1, 0, 2)
            ).tobytes())
        assert aligned.getvalue() == expected_data

    print("NVFP4 streamed pair-pack tests passed")


if __name__ == "__main__":
    main()
