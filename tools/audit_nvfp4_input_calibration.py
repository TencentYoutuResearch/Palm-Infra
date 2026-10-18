#!/usr/bin/env python3
"""Read only metadata/calibration scalars, never model weight payloads.

Exact bit equality is necessary (not sufficient) for sharing quantized inputs;
route membership and raw input identity still need separate validation.
"""
import argparse
import json
import math
import os
import struct

PACKAGE = struct.Struct("<II15Q")
WEIGHT = struct.Struct("<IIII8QII")
MAX_META = 256 * 1024 * 1024


def span(offset, length, size):
    if not isinstance(offset, int) or not isinstance(length, int):
        raise ValueError("noninteger span")
    if offset < 0 or length < 0 or offset > size or length > size - offset:
        raise ValueError("out-of-file span")


def read_at(source, offset, length, size):
    span(offset, length, size)
    source.seek(offset)
    data = source.read(length)
    if len(data) != length:
        raise ValueError("truncated file")
    return data


def audit(path):
    with open(path, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        header = PACKAGE.unpack(read_at(source, 0, PACKAGE.size, size))
        if header[:2] != (0x4D4C4F4D, 1):
            raise ValueError("unsupported package")
        meta_offset, meta_length = header[2:4]
        weight_offset, weight_length = header[12:14]
        if meta_offset < PACKAGE.size or not 0 < meta_length <= MAX_META:
            raise ValueError("invalid metadata size")
        span(weight_offset, weight_length, size)
        metadata = json.loads(read_at(source, meta_offset, meta_length, size))
        storage = metadata["moe_expert_storage"]
        rows = []
        payload_bytes = 0
        for layer in storage["layers"]:
            experts = layer.get("num_experts", storage.get("num_experts"))
            if not isinstance(experts, int) or not 0 < experts <= 65535:
                raise ValueError("invalid expert count")
            calibration = layer["gate_up"]["input_scales"]
            relative, length = metadata["weights"][calibration["weight"]]
            span(relative, length, weight_length)
            expected = experts * 2 * 4
            if (calibration["scales_per_expert"] != 2 or
                    calibration["weight_offset"] != relative or
                    calibration["data_bytes"] != expected or length < WEIGHT.size):
                raise ValueError("inconsistent calibration metadata")
            fields = WEIGHT.unpack(read_at(source, weight_offset + relative, WEIGHT.size, size))
            magic, flags, ndim, precision = fields[:4]
            if (magic, flags, ndim, precision) != (0x50414D58, 0, 2, 0):
                raise ValueError("invalid FP32 calibration header")
            if fields[4:6] != (experts, 2) or fields[9] != expected or fields[11]:
                raise ValueError("invalid calibration shape/size")
            data_offset = fields[8]
            if data_offset < WEIGHT.size or calibration["data_offset"] != data_offset:
                raise ValueError("inconsistent calibration data offset")
            span(data_offset, expected, length)
            data = read_at(source, weight_offset + relative + data_offset, expected, size)
            values = struct.unpack("<" + "f" * (experts * 2), data)
            if any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("nonpositive/nonfinite calibration")
            bits = struct.unpack("<" + "I" * (experts * 2), data)
            pairs = list(zip(bits[::2], bits[1::2]))
            equal = sum(gate == up for gate, up in pairs)
            rows.append({"layer": layer["layer"], "experts": experts,
                         "equal_gate_up_pairs": equal,
                         "unique_gate_globals": len(set(bits[::2])),
                         "unique_up_globals": len(set(bits[1::2])),
                         "unique_both_globals": len(set(bits)),
                         "unique_gate_up_pairs": len(set(pairs))})
            payload_bytes += expected
        if not rows or len({row["layer"] for row in rows}) != len(rows):
            raise ValueError("empty/duplicate layers")
        total = sum(row["experts"] for row in rows)
        return {"scope": "exact calibration-bit equality only, not input/route/TPS proof",
                "package": os.fspath(path), "layers": rows, "expert_pairs": total,
                "equal_gate_up_pairs": sum(row["equal_gate_up_pairs"] for row in rows),
                "calibration_payload_bytes_read": payload_bytes,
                "metadata_bytes_read": meta_length}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.package), indent=2))
