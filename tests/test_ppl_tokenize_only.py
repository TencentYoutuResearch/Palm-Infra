"""Tokenization must not load graphs, weights, or a compute backend."""
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = sys.argv.pop(1)


class TokenizeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mollm_tokenize_only_")
        self.addCleanup(self.tmp.cleanup)
        self.package = Path(self.tmp.name) / "sparse.mollm"
        metadata = json.dumps({"architecture": "test"}).encode()
        tokenizer = json.dumps({"model": {"vocab": {"a": 0, "b": 1}, "merges": []},
                                "added_tokens": [{"id": 2, "content": "<|begin_of_text|>"}]}).encode()
        # Intentionally invalid graphs and an 8 GiB sparse weight payload:
        # only section geometry, not model payloads, is relevant here.
        offset = 128
        sections = []
        for data in (metadata, tokenizer, b"", b"not-a-graph", b"not-a-graph"):
            sections.extend((offset, len(data)))
            offset += len(data)
        sections.extend((offset, 8 * 1024**3, 0, 0, 0))
        self.header = [0x4D4C4F4D, 1] + sections
        with self.package.open("wb") as target:
            target.write(struct.pack("<II15Q", *self.header))
            target.write(metadata + tokenizer + b"not-a-graph" * 2)
            target.truncate(offset + 8 * 1024**3)

    def run_case(self, *args):
        def bounded_address_space():
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", MOLLM_CUDA_HOST_HEADROOM_MB="bad")
        # Darwin rejects this RLIMIT_AS setting before exec. Keep all functional
        # checks there; Linux additionally proves the 8 GiB payload is not mapped
        # or copied under a 2 GiB address-space ceiling.
        limit_address_space = bounded_address_space if sys.platform.startswith("linux") else None
        return subprocess.run([BINARY, "--package", str(self.package), "--text", "ab",
                               "--tokenize-only", *args], env=environment, text=True,
                              capture_output=True, timeout=15, preexec_fn=limit_address_space)

    def test_sparse_weights_and_invalid_graphs_are_not_loaded(self):
        for device in ("cpu", "cuda", "metal"):
            with self.subTest(device=device):
                result = self.run_case("--device", device)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "tokens=2\ntoken_ids=0,1\n")
                self.assertNotIn("Backend:", result.stderr)
                self.assertNotIn("Engine:", result.stderr)

    def test_bos_and_cap_preserve_semantics(self):
        result = self.run_case("--prepend-bos", "--max-tokens", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "tokens=2\ntoken_ids=2,0\n")

    def test_qsa_tokenization_does_not_load_companion(self):
        result = self.run_case("--device", "cuda", "--qsa-companion", "missing-directory", "--n-ctx", "262144")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "tokens=2\ntoken_ids=0,1\n")
        self.assertNotIn("Backend:", result.stderr)
        self.assertNotIn("Engine:", result.stderr)

    def test_invalid_qsa_arguments_fail_before_model_loading(self):
        for args in (("--qsa-companion", "bundle"),
                     ("--device", "cuda", "--qsa-companion", ""),
                     ("--device", "cuda", "--qsa-companion", "bundle", "--n-ctx", "262145")):
            result = self.run_case(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--qsa-companion", result.stderr)
            self.assertNotIn("Backend:", result.stderr)
            self.assertNotIn("Engine:", result.stderr)

    def test_invalid_section_geometry_is_rejected(self):
        for field, value in ((0, 0), (1, 2), (4, 128), (5, 0), (4, 2**64 - 1)):
            with self.subTest(field=field, value=value):
                changed = self.header.copy()
                changed[field] = value
                with self.package.open("r+b") as target:
                    target.write(struct.pack("<II15Q", *changed))
                result = self.run_case()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
