"""Synthetic GDN target plus deliberately poor draft; no remote resources."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from run_tiny_qwen3_cpu_e2e import build_qwen35_fixture_model
from qwen35 import build_graph, export_weights, load_safetensors
from transpile import DimExpr, GraphBuilder, Precision, SEQ, save_package


def main():
    try:
        import numpy as np
    except ImportError:
        return 77
    with tempfile.TemporaryDirectory(prefix="mollm_recurrent_mtp_") as tmp:
        root = Path(tmp)
        model = root / "source"
        model.mkdir()
        build_qwen35_fixture_model(model)
        config = json.loads((model / "config.json").read_text())
        weights = load_safetensors(str(model / "model.safetensors"))
        config["text_config"]["linear_key_head_dim"] = 128
        config["text_config"]["linear_value_head_dim"] = 128
        rng = np.random.default_rng(20261005)
        prefix = "model.language_model.layers.0.linear_attn."
        for name, shape in {
            "in_proj_qkv.weight": (768, 32),
            "in_proj_z.weight": (256, 32),
            "out_proj.weight": (32, 256),
        }.items():
            weights[prefix + name] = rng.uniform(-0.1, 0.1, shape).astype(np.float16)
        weights[prefix + "conv1d.weight"] = rng.uniform(-0.05, 0.05, (768, 1, 4)).astype(np.float32)
        weights[prefix + "norm.weight"] = np.ones(128, dtype=np.float32)
        # Make recurrent contributions significant, rather than comparing a
        # nearly pure embedding residual with a vanishing state contribution.
        weights["model.language_model.layers.0.linear_attn.out_proj.weight"] *= 20
        export_weights(weights, str(root), config, quant="fp16")
        prefill = build_graph(".", config, seq_len=8, n_ctx=64, is_prefill=True)
        decode = build_graph(".", config, seq_len=1, n_ctx=64, is_prefill=False)
        mtp = GraphBuilder()
        hidden = mtp.input("target_hidden", (32, 8),
                           dynamic=(DimExpr.const(), DimExpr.seq()))
        mask = mtp.input("mask", (1, 8),
                         dynamic=(DimExpr.const(), DimExpr.seq()))
        ck = mtp.input("cache_k0", (8, 64, 4), prec=Precision.FP16)
        cv = mtp.input("cache_v0", (8, 64, 4), prec=Precision.FP16)
        qkv = mtp.permute(mtp.reshape(hidden, (8, 4, SEQ)), (0, 2, 1, 3))
        attn, _, _ = mtp.sdpa(qkv, qkv, qkv, mask, ck, cv,
                              num_heads=4, num_kv_heads=4,
                              head_dim=8, v_head_dim=8, scale=8 ** -0.5)
        output = mtp.reshape(mtp.permute(attn, (0, 2, 1, 3)), (32, SEQ))
        mtp.scalar_mul(output, -1.0)
        package = root / "recurrent_mtp.mollm"
        save_package(str(package), prefill, decode, str(root),
                     {"architecture": "qwen3_5", "hidden_size": 32,
                      "num_layers": 1, "n_ctx": 64, "prefill_seq_len": 8,
                      "vocab_size": 32}, g_mtp=mtp)
        subprocess.run([sys.argv[1], str(package)], check=True)
        # Load the same draft as an external companion, without
        # copying the target package or its shared embedding/head weights.
        target = root / "target_only.mollm"
        save_package(str(target), prefill, decode, str(root),
                     {"architecture": "qwen3_5", "hidden_size": 32,
                      "num_layers": 1, "n_ctx": 64, "prefill_seq_len": 8,
                      "vocab_size": 32})
        companion = root / "companion"
        companion.mkdir()
        mtp.save(str(companion / "model_mtp"))
        # Exercise both streamed manifest weights and node-local inline files.
        constant = next(node for node in mtp._nodes if node.weight_data is not None)
        original = companion / f"model_mtp_const_{constant.id}.weights"
        dedicated = companion / "mtp_scalar.weights"
        shutil.copyfile(original, dedicated)
        constant.weight_data = None
        constant.weight_path = "./mtp_scalar.weights"
        constant.params_str = [constant.weight_path]
        mtp.save(str(companion / "model_mtp"))
        manifest = {"format": "mollm-mtp-weight-bundle-v1", "standalone_model": False,
                    "graph": "model_mtp.graph", "seq_len": 8, "n_ctx": 64,
                    "shared_target_weights": ["./embed_tokens.weights", "./lm_head.weights"],
                    "weights": {"./mtp_scalar.weights": {"file": "mtp_scalar.weights",
                                 "bytes": dedicated.stat().st_size, "shape": [1], "precision": "FP32"}}}
        manifest_path = companion / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        subprocess.run([sys.argv[1], str(target), str(companion)], check=True)
        for invalid in ("size", "context", "escape", "extra", "precision", "shape"):
            bad = json.loads(json.dumps(manifest))
            entry = bad["weights"]["./mtp_scalar.weights"]
            if invalid == "size": entry["bytes"] += 1
            if invalid == "context": bad["n_ctx"] = 63
            if invalid == "escape": entry["file"] = "../mtp_scalar.weights"
            if invalid == "extra": bad["weights"]["./unused.weights"] = entry.copy()
            if invalid == "precision": entry["precision"] = "FP16"
            if invalid == "shape": entry["shape"] = [2]
            manifest_path.write_text(json.dumps(bad))
            subprocess.run([sys.argv[1], str(target), str(companion), "--reject-companion"], check=True)
        manifest_path.unlink()
        subprocess.run([sys.argv[1], str(target), str(companion), "--reject-companion"], check=True)

        # Separate wide recurrent MTP state from 32-wide logits hidden.
        # Replication is synthetic, but catches width/stride/shift plumbing.
        for graph in (prefill, decode):
            normal = graph._nodes[-1].id
            state = graph.tile(normal, (4, 1))
            graph.metadata["mtp_state_output_node"] = str(state)
            graph.scalar_mul(normal, 1.0)
        mtp = GraphBuilder()
        wide = mtp.input("target_hidden", (128, 8),
                         dynamic=(DimExpr.const(), DimExpr.seq()))
        hidden = mtp.slice_range(wide, offset=0, size=32, dim=0)
        mask = mtp.input("mask", (1, 8),
                         dynamic=(DimExpr.const(), DimExpr.seq()))
        ck = mtp.input("cache_k0", (8, 64, 4), prec=Precision.FP16)
        cv = mtp.input("cache_v0", (8, 64, 4), prec=Precision.FP16)
        qkv = mtp.permute(mtp.reshape(hidden, (8, 4, SEQ)), (0, 2, 1, 3))
        attn, _, _ = mtp.sdpa(qkv, qkv, qkv, mask, ck, cv,
                              num_heads=4, num_kv_heads=4,
                              head_dim=8, v_head_dim=8, scale=8 ** -0.5)
        output = mtp.scalar_mul(
            mtp.reshape(mtp.permute(attn, (0, 2, 1, 3)), (32, SEQ)), -1.0)
        state = mtp.tile(output, (4, 1))
        mtp.metadata["mtp_state_output_node"] = str(state)
        mtp.scalar_mul(output, 1.0)
        package = root / "wide_state_mtp.mollm"
        save_package(str(package), prefill, decode, str(root),
                     {"architecture": "qwen3_5", "hidden_size": 32,
                      "num_layers": 1, "n_ctx": 64, "prefill_seq_len": 8,
                      "vocab_size": 32}, g_mtp=mtp)
        subprocess.run([sys.argv[1], str(package)], check=True)
        wide_target = root / "wide_target_only.mollm"
        save_package(str(wide_target), prefill, decode, str(root),
                     {"architecture": "qwen3_5", "hidden_size": 32,
                      "num_layers": 1, "n_ctx": 64, "prefill_seq_len": 8,
                      "vocab_size": 32})
        wide_companion = root / "wide_companion"
        wide_companion.mkdir()
        mtp.save(str(wide_companion / "model_mtp"))
        wide_manifest = dict(manifest, weights={})
        (wide_companion / "manifest.json").write_text(json.dumps(wide_manifest))
        subprocess.run([sys.argv[1], str(wide_target), str(wide_companion)], check=True)
        # A 128-wide draft must not bind to a 32-wide target output.
        subprocess.run([sys.argv[1], str(target), str(wide_companion), "--reject-companion"], check=True)
        del mtp.metadata["mtp_state_output_node"]
        mtp.save(str(wide_companion / "model_mtp"))
        subprocess.run([sys.argv[1], str(wide_target), str(wide_companion), "--reject-companion"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
