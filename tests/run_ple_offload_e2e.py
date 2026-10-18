#!/usr/bin/env python3
"""Exercise resident PLE paging, numerical parity and failed-load cleanup."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run_tiny_qwen3_cpu_e2e import build_qwen3_fixture_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from qwen3 import build_graph, export_weights, load_safetensors
from transpile import Precision, StreamedWeight, WeightByteRange, save_package


def main():
    with tempfile.TemporaryDirectory(prefix="mollm_ple_e2e_") as tmp:
        root = Path(tmp)
        build_qwen3_fixture_model(root)
        cfg = json.loads((root / "config.json").read_text())
        export_weights(load_safetensors(str(root / "model.safetensors")), str(root))
        pf = build_graph(".", cfg, seq_len=4, n_ctx=8, is_prefill=True)
        dc = build_graph(".", cfg, seq_len=1, n_ctx=8)
        metadata = dict(cfg, architecture="qwen3", num_layers=1,
                        num_heads=4, num_kv_heads=1, n_ctx=8, prefill_seq_len=4)
        plain = root / "plain.mollm"
        save_package(str(plain), pf, dc, str(root), metadata)

        # An unused table isolates the loading policy from PLE arithmetic.
        # It is larger than the usual memlock limit, so pinning it by mistake
        # also makes the load fail on ordinary unprivileged test hosts.
        raw = root / "ple.bin"
        size = 16 * 1024 * 1024
        with raw.open("wb") as f:
            f.truncate(size)
        ref = "./model_language_model_layers_0_ple_ple_embedding_ngram_embedding_weight.weights"
        for graph in (pf, dc):
            graph.weight(ref, (1024, size // 1024), Precision.RAW_U8)
        ple = root / "ple.mollm"
        save_package(str(ple), pf, dc, str(root), metadata, streamed_weights={
            ref: StreamedWeight(Precision.RAW_U8, (1024, size // 1024),
                                [WeightByteRange(str(raw), 0, size)])})
        subprocess.run([sys.argv[1], "--ple-offload", str(ple), str(plain)], check=True)


if __name__ == "__main__":
    main()
