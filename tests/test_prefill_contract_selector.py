"""Early, fail-closed engine guard checks; never open a real model package."""
import os
import subprocess
import sys
import unittest

BINARY = sys.argv.pop(1)
BASE = [BINARY, "--package", "/nonexistent-cuda-prefill-test.mollm",
        "--device", "cuda", "--offload-ple", "--cuda-prefill-reclaim",
        "--cuda-prefill-logical-batch",
        "--n-ctx", "2048", "--prefill-chunk-size", "2048",
        "--max-new-tokens", "1"]
OPTIONS = {
    "MOLLM_RESIDENT_EXPERT_ARENA": "cuda-device",
    "MOLLM_CUDA_HOST_HEADROOM_MB": "24576",
    "MOLLM_CUDA_OUTPUT_POOL_MB": "2048",
    "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS": "1",
}


class GuardTests(unittest.TestCase):
    def run_case(self, changes=None, args=None, require_headroom=True):
        environment = os.environ.copy()
        environment.update(OPTIONS)
        for name, value in (changes or {}).items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        result = subprocess.run(args or BASE, env=environment, capture_output=True,
                                text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)  # Missing package even if guard accepts.
        # The CLI watchdog runs before engine guards. Small CI hosts cannot
        # reserve the 24 GiB required by logical prefill.
        if require_headroom and "CUDA host watchdog: cannot start" in result.stderr:
            self.assertNotIn("MoE: loading", result.stderr)
            self.skipTest("host cannot reserve the required CUDA memory headroom")
        return result.stderr

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux CLI memory watchdog")
    def test_host_watchdog_rejects_impossible_headroom(self):
        error = self.run_case({"MOLLM_CUDA_HOST_HEADROOM_MB": "1073741824"},
                              require_headroom=False)
        self.assertIn("CUDA host watchdog: cannot start", error)
        self.assertNotIn("CudaBackend:", error)
        self.assertNotIn("MoE: loading", error)

    def test_invalid_environment_rejected_before_backend(self):
        cases = {
            "MOLLM_RESIDENT_EXPERT_ARENA": (None, "managed"),
            "MOLLM_CUDA_HOST_HEADROOM_MB": (None, "", "0", "24575", "-1", "+24576", " 24576", "24576x", "184467440737095516160"),
            "MOLLM_CUDA_OUTPUT_POOL_MB": (None, "", "0", "4097", "-1", "2048x", "184467440737095516160"),
            "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS": (None,),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    error = self.run_case({name: value})
                    if name == "MOLLM_CUDA_HOST_HEADROOM_MB" and value not in (None, "0", "24575"):
                        self.assertIn("CUDA host watchdog: invalid MOLLM_CUDA_HOST_HEADROOM_MB", error)
                    else:
                        self.assertIn("--cuda-prefill-logical-batch requires", error)
                    self.assertNotIn("CudaBackend:", error)

    def test_mtp_padding_and_missing_public_reclaim_rejected(self):
        variants = [BASE + ["--mtp-draft-tokens", "1"],
                    BASE + ["--static-padded"],
                    [arg for arg in BASE if arg != "--cuda-prefill-reclaim"]]
        for args in variants:
            with self.subTest(args=args):
                error = self.run_case(args=args)
                self.assertIn("--cuda-prefill-logical-batch requires", error)
                self.assertNotIn("CudaBackend:", error)

    def test_invalid_context_and_chunk_rejected(self):
        for option, value, message in (
            ("--n-ctx", "0", "invalid value for --n-ctx"),
            ("--prefill-chunk-size", "2049", "--cuda-prefill-logical-batch requires"),
            ("--device", "cpu", "--cuda-prefill-logical-batch requires"),
        ):
            bad = BASE.copy()
            bad[bad.index(option) + 1] = value
            error = self.run_case(args=bad)
            self.assertIn(message, error)

    def test_valid_setup_reaches_bounded_prefill(self):
        error = self.run_case()
        if "CUDA backend unavailable" in error or "built without MOLLM_CUDA" in error:
            self.skipTest("CUDA backend unavailable")
        self.assertIn("sequence cap2048", error)
        self.assertIn("prefill stateful read-only last-use contracts enabled", error)
        self.assertNotIn("MoE: loading", error)

    def test_context_and_chunk_are_independent(self):
        # These guards run before opening the nonexistent package. No model
        # allocation or inference is needed to verify the public contract.
        # A chunk is an upper bound, not a minimum prompt/context length.
        for context in ("1", "1024", "2049", "16384", "32768"):
            for chunk in ("0", "1", "128", "957", "1024", "2047", "2048"):
                with self.subTest(context=context, chunk=chunk):
                    args = BASE.copy()
                    args[args.index("--n-ctx") + 1] = context
                    args[args.index("--prefill-chunk-size") + 1] = chunk
                    error = self.run_case(args=args)
                    if "CUDA backend unavailable" in error or "built without MOLLM_CUDA" in error:
                        self.skipTest("CUDA backend unavailable")
                    self.assertNotIn("--cuda-prefill-logical-batch requires", error)
                    self.assertIn("sequence cap2048", error)
                    self.assertNotIn("MoE: loading", error)


if __name__ == "__main__":
    unittest.main()
