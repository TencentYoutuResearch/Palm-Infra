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
        self.root = Path(self.tmp.name)/"palm-infra"
        self.root.mkdir()
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
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
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
        result = self.run_case("--dense-bringup", "--n-ctx", "262144")
        self.assertEqual(self.chunk(result), ["2048"])
        self.assertNotIn("--qsa-companion", result["args"])
        self.assertNotIn("--dense-bringup", result["args"])
        result = self.run_case("--qsa-companion", "bundle", "--n-ctx", "not-an-int")
        self.assertIn("not-an-int", result["args"])
        self.assertEqual(self.chunk(result), ["2048"])

    def test_default_qsa_and_package_overrides(self):
        result = self.run_case()
        self.assertIn("--qsa-companion", result["args"])
        self.assertIn("../qwen38_qsa_bundle", result["args"])
        result = self.run_case("--n-ctx", "262144")
        self.assertEqual(self.chunk(result), ["512"])
        result = self.run_case("--package", "another.mollm", "--n-ctx", "262144")
        self.assertNotIn("--qsa-companion", result["args"])
        self.assertEqual(self.chunk(result), ["2048"])
        result = self.run_case("--package", "another.mollm", "--package", "../qwen38_flash_next_nvfp4_calibrated.mollm")
        self.assertIn("--qsa-companion", result["args"])

    def test_package_identity_without_gnu_realpath(self):
        from unittest.mock import patch
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        realpath = bin_dir / "realpath"
        realpath.write_text("#!/bin/sh\nexit 1\n")
        realpath.chmod(0o755)
        default = self.root.parent / "qwen38_flash_next_nvfp4_calibrated.mollm"
        with patch.dict(os.environ, {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}):
            # Help must work even before the default package has been created.
            for path in (str(default), "./../" + default.name):
                self.assertIn("--qsa-companion", self.run_case("--package", path)["args"])
            for path in ("another.mollm", "missing/" + default.name):
                self.assertNotIn("--qsa-companion", self.run_case("--package", path)["args"])
            default.touch()
            alias = self.root / "model-alias.mollm"
            alias.symlink_to(default)
            self.assertIn("--qsa-companion", self.run_case("--package", str(alias))["args"])
            other = self.root / default.name
            other.touch()
            self.assertNotIn("--qsa-companion", self.run_case("--package", str(other))["args"])

    def test_prompt_text_is_not_a_flag(self):
        result = self.run_case("--dense-bringup", "--prompt", "--qsa-companion")
        self.assertEqual(result["args"][-2:], ["--prompt", "--qsa-companion"])
        self.assertEqual(result["args"].count("--qsa-companion"), 1)
        result = self.run_case("--prompt", "--dense-bringup", "--n-ctx", "262144")
        self.assertEqual(self.chunk(result), ["512"])
        self.assertIn("--dense-bringup", result["args"])

    def test_missing_bundle_rejects_before_gpu_checks(self):
        result = subprocess.run(["bash", str(self.root/SCRIPT.name)], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing QSA bundle", result.stderr)

    def test_explicit_dense_conflict(self):
        result = subprocess.run(["bash", str(self.root/SCRIPT.name), "--dense-bringup", "--qsa-companion", "bundle"],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflicts", result.stderr)

if __name__ == "__main__":
    unittest.main()
