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
                        "print(json.dumps(dict(args=sys.argv[1:],cache=os.environ['MOLLM_CUDA_PREFILL_FP16_CACHE_MB'],segments_disabled=os.environ.get('MOLLM_CUDA_DISABLE_DECODE_SEGMENTS'),policy={k:v for k,v in os.environ.items() if k.startswith('MOLLM_')})))\n")
        chat.chmod(0o755)

    def run_case(self, *args, cache=None, segments_disabled=None, overrides=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOLLM_")}
        if segments_disabled is not None:
            env["MOLLM_CUDA_DISABLE_DECODE_SEGMENTS"] = segments_disabled
        if cache is not None:
            env["MOLLM_CUDA_PREFILL_FP16_CACHE_MB"] = cache
        env.update(overrides or {})
        # --help deliberately bypasses all GPU/process/memory checks.
        result = subprocess.run(["bash", str(self.root / SCRIPT.name), "--help", *args],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def chunk(self, result):
        return [result["args"][i+1] for i, arg in enumerate(result["args"][:-1]) if arg == "--prefill-chunk-size"]

    def run_mtp_case(self, *args):
        result = self.run_case("--mtp", *args)
        self.assertNotIn("--mtp", result["args"])
        self.assertEqual(result["cache"], "2048")
        return result

    def value(self, result, flag):
        args = result["args"]
        return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == flag][-1]

    def test_environment_is_limited_to_resource_settings(self):
        allowed = {"MOLLM_RESIDENT_EXPERT_ARENA", "MOLLM_CUDA_PREFILL_FP16_CACHE_MB",
                   "MOLLM_CUDA_HOST_HEADROOM_MB", "MOLLM_CUDA_OUTPUT_POOL_MB",
                   "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS"}
        for data in (self.run_case(), self.run_mtp_case()):
            self.assertLessEqual(set(data["policy"]), allowed)

    def test_mtp_preset(self):
        result = self.run_mtp_case()
        self.assertEqual(self.value(result, "--mtp-companion"), "../qwen38_mtp_qsa_bundle")
        self.assertEqual(self.value(result, "--mtp-draft-tokens"), "2")
        self.assertEqual(self.value(result, "--temperature"), "0")
        self.assertEqual(self.value(result, "--n-ctx"), "32768")
        # Cached verification workspaces need an output pool of at most 3 GiB.
        self.assertEqual(result["policy"]["MOLLM_CUDA_OUTPUT_POOL_MB"], "3072")
        self.assertIsNone(result["segments_disabled"])
        overridden = self.run_mtp_case("--mtp-draft-tokens", "3")
        self.assertEqual(self.value(overridden, "--mtp-draft-tokens"), "3")
        self.assertEqual(self.run_case()["policy"]["MOLLM_CUDA_OUTPUT_POOL_MB"], "4096")

    def test_mtp_rejects_dense_and_custom_packages(self):
        for args in (("--mtp", "--dense-bringup"), ("--mtp", "--package", "custom.mollm")):
            result = subprocess.run(["bash", str(self.root / SCRIPT.name), "--help", *args],
                                    env={k: v for k, v in os.environ.items()
                                         if not k.startswith("MOLLM_")},
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_qualified_decode_segment_policy(self):
        for context in ("2048", "16384", "32768"):
            self.assertIsNone(self.run_case("--n-ctx", context)["segments_disabled"])
        for value in ("1", "0", ""):
            # The backend tests presence: even zero/empty intentionally disable.
            self.assertEqual(self.run_case(segments_disabled=value)["segments_disabled"], value)
        for args in (("--n-ctx", "32769"), ("--dense-bringup",),
                     ("--package", "custom.mollm", "--qsa-companion", "bundle")):
            self.assertEqual(self.run_case(*args)["segments_disabled"], "1")

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
