#!/usr/bin/env python3
"""Export a local MTP-only graph/weight bundle, not a standalone model package."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from qwen4_exp import build_mtp_graph, build_mtp_streamed_weights
from transpile import _read_weight_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--refresh-graph", action="store_true",
                        help="Regenerate only graph/inline scalars in an existing validated bundle; never rewrite model weights")
    args = parser.parse_args()
    if not args.refresh_graph and args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("output directory must be new or empty; existing files are never overwritten")
    cfg = json.loads((args.model_dir / "config.json").read_text())
    graph = build_mtp_graph(".", cfg, args.seq_len, args.n_ctx)
    if args.refresh_graph:
        manifest = json.loads((args.output_dir / "manifest.json").read_text())
        if (manifest["format"] != "mollm-mtp-weight-bundle-v1" or
                manifest["source_model_dir"] != str(args.model_dir.resolve()) or
                manifest["seq_len"] != args.seq_len or manifest["n_ctx"] != args.n_ctx):
            raise ValueError("existing bundle does not match requested source/graph geometry")
        refs = {node.weight_path for node in graph._nodes if node.weight_path}
        shared = set(manifest["shared_target_weights"])
        if refs - shared != set(manifest["weights"]):
            raise ValueError("existing graph/weight reference mismatch")
        for node in graph._nodes:
            if not node.weight_path or node.weight_path in shared:
                continue
            entry = manifest["weights"][node.weight_path]
            if node.weight_path != "./" + entry["file"] or Path(entry["file"]).name != entry["file"]:
                raise ValueError("invalid existing weight filename")
            path = args.output_dir / entry["file"]
            header = _read_weight_header(path)
            shape = tuple(header["shape"])
            wanted = tuple(node.out_shape) + (1,) * (4 - len(node.out_shape))
            if (path.stat().st_size != entry["bytes"] or
                    header["precision"] != int(node.out_prec) or shape != wanted):
                raise ValueError(f"existing weight geometry/precision mismatch: {path.name}")
        graph.save(str(args.output_dir / "model_mtp"))
        print("Refreshed graph/inline constants only; all streamed model weight files unchanged.")
        return
    weights = build_mtp_streamed_weights(args.model_dir, cfg)
    shared = {"./embed_tokens.weights", "./lm_head.weights"}
    refs = {node.weight_path for node in graph._nodes if node.weight_path}
    if refs - shared != set(weights):
        raise ValueError("MTP graph/weight reference mismatch")
    for node in graph._nodes:
        if node.weight_path in weights and weights[node.weight_path].precision != node.out_prec:
            raise ValueError(f"MTP graph/weight precision mismatch: {node.weight_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest = {"format": "mollm-mtp-weight-bundle-v1", "standalone_model": False,
                "shared_target_weights": sorted(shared), "graph": "model_mtp.graph",
                "source_model_dir": str(args.model_dir.resolve()),
                "seq_len": args.seq_len, "n_ctx": args.n_ctx, "weights": {}}
    for ref, weight in weights.items():
        name = Path(ref).name
        if ref != "./" + name or not name.endswith(".weights"):
            raise ValueError("unexpected MTP output filename")
        with (args.output_dir / name).open("xb") as output:
            weight.write_to(output)
        manifest["weights"][ref] = {"file": name, "bytes": weight.size,
                                     "shape": weight.logical_shape,
                                     "precision": weight.precision.name}
        print(f"wrote {name} bytes={weight.size}", flush=True)
    graph.save(str(args.output_dir / "model_mtp"))
    manifest["elapsed_seconds"] = time.monotonic() - start
    # Manifest is written last; a failed conversion leaves an inspectable
    # partial directory without claiming the bundle is complete.
    with (args.output_dir / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2)
    print(f"bundle={args.output_dir} elapsed_seconds={manifest['elapsed_seconds']:.2f}", flush=True)
    print("Not standalone: load with a matching target package using --mtp-companion and --mtp-draft-tokens.")


if __name__ == "__main__":
    main()
