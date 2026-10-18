"""QSA-only streamed export and centered norm semantics, no GPU/model load."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from export_qwen4_qsa_weights import export_bundle
from transpile import _read_weight_header


def fixture(root, bad_shape=False):
    cfg = dict(hidden_size=8, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
               indexer_n_heads=2, indexer_head_dim=4, indexer_kv_heads=1, indexer_compress_ratio=4,
               indexer_budget=8, head_dim=8, num_attention_heads=2, num_key_value_heads=1,
               rope_parameters=dict(rope_theta=10000000, partial_rotary_factor=.5))
    (root / "config.json").write_text(json.dumps(cfg))
    prefix = "model.language_model.layers.1.self_attn.indexer."
    tensors = {prefix + "index_qk_proj.weight": np.arange(12 * (7 if bad_shape else 8), dtype=np.float32).reshape(12, -1) / 16,
               prefix + "q_layernorm.weight": np.full(4, -.125, dtype=np.float32),
               prefix + "k_layernorm.weight": np.full(4, .25, dtype=np.float32)}
    header, payload, offset = {}, [], 0
    for name, value in tensors.items():
        data = (value.view(np.uint32) >> 16).astype('<u2').tobytes()
        header[name] = dict(dtype="BF16", shape=list(value.shape), data_offsets=[offset, offset + len(data)])
        payload.append(data);offset += len(data)
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(payload))
    return tensors


class QsaBundleTest(unittest.TestCase):
    def test_small_bundle_does_not_add_one_or_rewrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);tensors = fixture(root)
            source_hash = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
            out = root / "bundle"
            manifest = export_bundle(root, out)
            self.assertFalse(manifest["standalone_model"])
            self.assertEqual(manifest["norm_weight_format"], "centered")
            self.assertEqual(manifest["geometry"]["layers"], [1])
            self.assertEqual(len(manifest["weights"]), 3)
            for ref, entry in manifest["weights"].items():
                path = out / entry["file"];header = _read_weight_header(path)
                data = path.read_bytes()[header["data_offset"]:header["data_offset"] + header["data_size"]]
                actual = np.frombuffer(data, '<f2' if entry["precision"] == "FP16" else '<f4')
                original = next(value for name, value in tensors.items()
                                if name.replace('.', '_') + '.weights' == Path(ref).name)
                np.testing.assert_array_equal(actual, original.ravel())
                self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(source_hash, hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest())
            before = (out / "manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                export_bundle(root, out)
            self.assertEqual(before, (out / "manifest.json").read_bytes())

    def test_bad_geometry_leaves_no_completion_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);fixture(root, bad_shape=True)
            with self.assertRaises(ValueError):
                export_bundle(root, root / "bundle")
            self.assertFalse((root / "bundle").exists())


if __name__ == '__main__':
    unittest.main()
