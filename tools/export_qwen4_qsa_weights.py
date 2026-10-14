#!/usr/bin/env python3
"""Export only missing QSA indexer weights; never rewrite the target package.

The bundle contains weights/geometry, not a standalone model or a graph.
Its runtime companion loader is a separate integration step.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from safetensors_stream import SafeTensorIndex, dense_streamed_weight, fp32_streamed_weight
from transpile import Precision, _read_weight_header


def descriptors(model_dir, cfg):
    tc = cfg.get("text_config", cfg)
    heads = int(tc.get("indexer_n_heads", 4))
    dim = int(tc.get("indexer_head_dim", 128))
    ratio = int(tc.get("indexer_compress_ratio", 4))
    budget = int(tc.get("indexer_budget", 2048))
    hidden = int(tc["hidden_size"])
    rope = tc["rope_parameters"]
    rope_dim = int(int(tc["head_dim"]) * float(rope["partial_rotary_factor"]))
    theta, eps = float(rope["rope_theta"]), float(tc.get("rms_norm_eps", 1e-6))
    layers = [i for i, kind in enumerate(tc["layer_types"]) if kind == "full_attention"]
    if (not layers or len(tc["layer_types"]) != int(tc["num_hidden_layers"]) or
            int(tc.get("indexer_kv_heads", 1)) != 1 or not 1 <= heads <= 16 or
            not 2 <= rope_dim <= dim <= 256 or rope_dim % 2 or hidden < 1 or
            not 1 <= ratio <= 16 or budget < 1 or budget % ratio or budget + ratio - 1 > 4096 or
            not math.isfinite(theta) or theta <= 0 or not math.isfinite(eps) or eps <= 0):
        raise ValueError("unsupported QSA geometry")
    source = SafeTensorIndex(model_dir)
    weights = {}
    for layer in layers:
        prefix = f"model.language_model.layers.{layer}.self_attn.indexer"
        output = f"model_language_model_layers_{layer}_self_attn_indexer"
        for name, shape in (("index_qk_proj", ((heads + 1) * dim, hidden)),
                            ("q_layernorm", (dim,)), ("k_layernorm", (dim,))):
            original = f"{prefix}.{name}.weight"
            tensor = source.tensor(original)
            if tensor.shape != shape:
                raise ValueError(f"QSA weight shape mismatch: {original}")
            if name == "index_qk_proj":
                if tensor.dtype not in ("BF16", "F16"):
                    raise ValueError("QSA projection currently requires BF16/FP16 source")
                weight = dense_streamed_weight(source, original)
            else:
                # ORIGINAL centered weights, not fp32_plus_one_streamed_weight.
                weight = fp32_streamed_weight(source, original)
            weights[f"./{output}_{name}_weight.weights"] = weight
    geometry = dict(hidden_size=hidden, heads=heads, dimension=dim, ratio=ratio,
                    budget=budget, rope_dim=rope_dim, rope_theta=theta, eps=eps,
                    layers=layers, main_heads=int(tc["num_attention_heads"]),
                    main_kv_heads=int(tc["num_key_value_heads"]), main_dimension=int(tc["head_dim"]))
    return geometry, weights


def export_bundle(model_dir, output_dir):
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty; existing files are never overwritten")
    config_bytes = (model_dir / "config.json").read_bytes()
    geometry, weights = descriptors(model_dir, json.loads(config_bytes))
    manifest = dict(format="mollm-qsa-weight-bundle-v1", standalone_model=False,
                    activation_precision="FP32", norm_weight_format="centered",
                    source_model_dir=str(model_dir.resolve()),
                    source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    geometry=geometry, weights={})
    output_dir.mkdir(parents=True, exist_ok=True)
    for ref, weight in weights.items():
        path = output_dir / Path(ref).name
        with path.open("xb") as target:
            weight.write_to(target)
        header = _read_weight_header(path)
        if path.stat().st_size != weight.size or header["precision"] != int(weight.precision):
            raise ValueError(f"written weight contract mismatch: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as target:
            for chunk in iter(lambda: target.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["weights"][ref] = dict(file=path.name, bytes=weight.size,
                                        shape=weight.logical_shape, precision=weight.precision.name,
                                        sha256=digest.hexdigest())
    # Completion manifest last; interrupted export leaves a partial directory.
    with (output_dir / "manifest.json").open("x") as target:
        json.dump(manifest, target, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = export_bundle(args.model_dir, args.output_dir)
    size = sum(entry["bytes"] for entry in manifest["weights"].values())
    print(f"QSA bundle: {len(manifest['weights'])} weights, {size} bytes, {args.output_dir}")
    print("Not standalone; no target-package rewrite. Runtime companion loading is not yet implemented.")


if __name__ == "__main__":
    main()
