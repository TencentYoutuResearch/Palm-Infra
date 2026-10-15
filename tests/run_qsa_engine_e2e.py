"""Small synthetic Qwen4Exp QSA model, not a checkpoint quality/performance gate."""
import sys
import tempfile
from pathlib import Path
import subprocess
import hashlib
import json
try:
    import numpy as np
except ImportError:
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from qwen4_exp import build_graph
from transpile import Precision, save_package, _write_weight_file


def main():
    tc = dict(hidden_size=32, num_hidden_layers=1, layer_types=["full_attention"],
              rms_norm_eps=1e-6, hc_count=4, hc_lowrank=8, indexer_budget=8,
              indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_compress_ratio=4,
              rope_parameters=dict(rope_theta=1000000, partial_rotary_factor=.5),
              head_dim=8, num_attention_heads=4, num_key_value_heads=2,
              linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=4,
              linear_value_head_dim=4, linear_conv_kernel_dim=4, moe_intermediate_size=8,
              shared_expert_intermediate_size=8, num_experts=8, num_experts_per_tok=2,
              vocab_size=64, ple_layer_ids=[], eos_token_id=63)
    # Package ctx32, runtime ctx64: auxiliary states must resize with main KV.
    pf = build_graph(".", tc, seq_len=17, n_ctx=32, is_prefill=True, enable_qsa=True)
    dc = build_graph(".", tc, seq_len=1, n_ctx=32, enable_qsa=True)
    with tempfile.TemporaryDirectory(prefix="mollm_qsa_engine_") as directory:
        root = Path(directory)
        rng = np.random.default_rng(91273)
        written = set()
        for graph in (pf, dc):
            for node in graph._nodes:
                if not node.weight_path or node.weight_path in written:
                    continue
                written.add(node.weight_path)
                shape = tuple(node.out_shape[:2]) if node.out_shape[1] != 1 else (node.out_shape[0],)
                if node.out_prec == Precision.FP32:
                    data = np.zeros(shape, dtype=np.float32) if "indexer" in node.weight_path else np.ones(shape, dtype=np.float32)
                else:
                    data = rng.normal(0, .025, shape).astype(np.float16)
                _write_weight_file(str(root / Path(node.weight_path).name), data)
        metadata = dict(architecture="qwen4-exp", hidden_size=32, num_layers=1,
                        num_heads=4, num_kv_heads=2, head_dim=8, vocab_size=64,
                        n_ctx=32, prefill_seq_len=17, rope_dim=4, rope_theta=1000000)
        package = root / "qsa.mollm"
        save_package(str(package), pf, dc, str(root), metadata)
        # Same main weights, legacy dense graph, missing indexer supplied separately.
        dense_tc = dict(tc, indexer_budget=64)
        dense_pf = build_graph(".", dense_tc, seq_len=17, n_ctx=32, is_prefill=True)
        dense_dc = build_graph(".", dense_tc, seq_len=1, n_ctx=32)
        dense_package = root / "dense.mollm"
        save_package(str(dense_package), dense_pf, dense_dc, str(root), metadata)
        bundle = root / "companion"
        bundle.mkdir()
        manifest = dict(format="mollm-qsa-weight-bundle-v1", standalone_model=False,
                        activation_precision="FP32", norm_weight_format="centered",
                        geometry=dict(hidden_size=32, heads=2, dimension=8, ratio=4, budget=8,
                                      rope_dim=4, rope_theta=1000000, eps=1e-6, layers=[0],
                                      main_heads=4, main_kv_heads=2, main_dimension=8), weights={})
        for graph_node in pf._nodes:
            if not graph_node.weight_path or "self_attn_indexer_" not in graph_node.weight_path:
                continue
            basename = Path(graph_node.weight_path).name
            payload = (root / basename).read_bytes()
            (bundle / basename).write_bytes(payload)
            shape = list(graph_node.out_shape[:2]) if graph_node.out_shape[1] != 1 else [graph_node.out_shape[0]]
            manifest["weights"]["./" + basename] = dict(file=basename, bytes=len(payload), shape=shape,
                                                      precision=graph_node.out_prec.name,
                                                      sha256=hashlib.sha256(payload).hexdigest())
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        result = subprocess.run([sys.argv[1], str(package), str(dense_package), str(bundle)])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        # Failure paths must not silently revert to dense attention.
        reference = next(iter(manifest["weights"]))
        for mutate in (lambda m: m["weights"][reference].update(sha256="0" * 64),
                       lambda m: m["weights"][reference].update(file="../escape.weights"),
                       lambda m: m["geometry"].update(main_heads=8),
                       lambda m: m["geometry"].update(rope_dim=2),
                       lambda m: m["geometry"].update(layers=[0,0])):
            invalid = json.loads(json.dumps(manifest))
            mutate(invalid)
            (bundle / "manifest.json").write_text(json.dumps(invalid))
            subprocess.run([sys.argv[1], str(package), str(dense_package), str(bundle), "--reject"], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
