#!/usr/bin/env python3
"""Exercise engine package calibration validation with unused tiny experts."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from run_tiny_qwen3_cpu_e2e import build_qwen3_fixture_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from qwen3 import build_graph, export_weights, load_safetensors
from transpile import Precision, StreamedWeight, WeightByteRange, save_package


def main():
    with tempfile.TemporaryDirectory(prefix="mollm_nvfp4_calibration_") as tmp:
        root = Path(tmp)
        build_qwen3_fixture_model(root)
        cfg = json.loads((root / "config.json").read_text())
        export_weights(load_safetensors(str(root / "model.safetensors")), str(root))
        pf = build_graph(".", cfg, seq_len=4, n_ctx=8, is_prefill=True)
        dc = build_graph(".", cfg, seq_len=1, n_ctx=8)
        metadata = dict(cfg, architecture="qwen3", num_layers=1,
                        num_heads=4, num_kv_heads=1, n_ctx=8, prefill_seq_len=4)
        raw = root / "expert.bin"
        raw.write_bytes(bytes([0x63]) * 64 + bytes([0x38]) * 8 + struct.pack("<4f", *([1.] * 4)))
        inputs = root / "inputs.bin"
        inputs.write_bytes(struct.pack("<3f", .0013892765, .0031854538, .002))
        streamed = {}
        layer = {"layer": 0, "num_experts": 1}
        for name, count, offset in (("gate_up", 2, 0), ("down", 1, 8)):
            ref, calibration_ref = f"./{name}.weights", f"./{name}_input.weights"
            streamed[ref] = StreamedWeight(Precision.NVFP4, (4, 32),
                [WeightByteRange(str(raw), 0, 64)], [WeightByteRange(str(raw), 64, 24)],
                group_size=16, num_groups=8)
            streamed[calibration_ref] = StreamedWeight(Precision.FP32, (1, count),
                [WeightByteRange(str(inputs), offset, count * 4)])
            for graph in (pf, dc):
                graph.weight(ref, (4, 32), Precision.NVFP4)
                graph.weight(calibration_ref, (1, count), Precision.FP32)
            layer[name] = {"weight": ref, "rows_per_expert": 4, "cols": 32,
                           "input_scales_weight": calibration_ref}
        metadata["moe_expert_storage"] = {"num_experts": 1, "layers": [layer]}
        valid = root / "valid.mollm"
        save_package(str(valid), pf, dc, str(root), metadata, streamed_weights=streamed)
        payload = valid.read_bytes()
        header = struct.unpack("<II15Q", payload[:128])
        meta = json.loads(payload[header[2]:header[2] + header[3]])
        calibration = meta["moe_expert_storage"]["layers"][0]["gate_up"]["input_scales"]
        base = header[12] + calibration["weight_offset"]
        mutations = [(base + calibration["data_offset"], struct.pack("<f", value))
                     for value in (0., -1., float("inf"), float("nan"))]
        # XMAP precision, first shape dimension, and data offset.
        mutations.extend([(base + 12, struct.pack("<I", int(Precision.FP16))),
                          (base + 16, struct.pack("<Q", 2)),
                          (base + 48, struct.pack("<Q", 2**63))])
        bad_paths = []
        for i, (offset, replacement) in enumerate(mutations):
            bad = bytearray(payload)
            bad[offset:offset + len(replacement)] = replacement
            path = root / f"invalid-{i}.mollm"
            path.write_bytes(bad)
            bad_paths.append(str(path))
        subprocess.run([sys.argv[1], "--nvfp4-calibration", str(valid), *bad_paths], check=True)


if __name__ == "__main__":
    main()
