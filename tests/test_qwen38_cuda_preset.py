"""Inspect preset argument/env selection with a fake chat; never touch a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "chat-qwen38-cuda.sh"


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mollm_preset_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copyfile(SCRIPT, self.root / SCRIPT.name)
        (self.root / "build_cuda").mkdir()
        chat = self.root / "build_cuda/mollm_chat"
        chat.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                        "print(json.dumps(dict(args=sys.argv[1:],cache=os.environ['MOLLM_CUDA_PREFILL_FP16_CACHE_MB'])))\n")
        chat.chmod(0o755)

    def run_case(self, *args, cache=None):
        env = os.environ.copy()
        env.pop("MOLLM_CUDA_PREFILL_FP16_CACHE_MB", None)
        if cache is not None:
            env["MOLLM_CUDA_PREFILL_FP16_CACHE_MB"] = cache
        # --help deliberately bypasses all GPU/process/memory checks.
        result = subprocess.run(["bash", str(self.root / SCRIPT.name), "--help", *args],
                                env=env, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def chunk(self, result):
        return [result["args"][i+1] for i, arg in enumerate(result["args"][:-1]) if arg == "--prefill-chunk-size"]

    def test_existing_small_context_unchanged(self):
        for args in ((), ("--qsa-companion", "bundle", "--n-ctx", "65536")):
            result = self.run_case(*args)
            self.assertEqual(result["cache"], "8192")
            self.assertEqual(self.chunk(result), ["2048"])

    def test_large_qsa_context_defaults(self):
        for context in ("65537", "131072", "262144", "000262144", "+262144"):
            result = self.run_case("--qsa-companion", "bundle", "--n-ctx", context)
            self.assertEqual(result["cache"], "4096")
            self.assertEqual(self.chunk(result), ["512"])

    def test_explicit_overrides_and_last_context(self):
        result = self.run_case("--qsa-companion", "bundle", "--n-ctx", "262144",
                               "--prefill-chunk-size", "1024", cache="2048")
        self.assertEqual(result["cache"], "2048")
        self.assertEqual(self.chunk(result), ["512", "1024"])
        result = self.run_case("--qsa-companion", "bundle", "--n-ctx", "262144", "--n-ctx", "16384")
        self.assertEqual(result["cache"], "8192")
        self.assertEqual(self.chunk(result), ["2048"])

    def test_dense_and_invalid_arguments_not_reinterpreted(self):
        result = self.run_case("--n-ctx", "262144")
        self.assertEqual(self.chunk(result), ["2048"])
        result = self.run_case("--qsa-companion", "bundle", "--n-ctx", "not-an-int")
        self.assertIn("not-an-int", result["args"])
        self.assertEqual(self.chunk(result), ["2048"])


if __name__ == "__main__":
    unittest.main()
