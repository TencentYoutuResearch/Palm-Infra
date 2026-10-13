"""Early, fail-closed engine guard checks; never open a real model package."""
import os
import subprocess
import sys
import unittest

BINARY = sys.argv.pop(1)
BASE = [BINARY, "--package", "/nonexistent-stateful-contract-test.mollm",
        "--device", "cuda", "--offload-ple", "--cuda-prefill-reclaim",
        "--n-ctx", "4096", "--prefill-chunk-size", "256", "--max-new-tokens", "1"]
OPTIONS = {
    "MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE": "1",
    "MOLLM_RESIDENT_EXPERT_ARENA": "cuda-device",
    "MOLLM_CUDA_HOST_HEADROOM_MB": "24576",
    "MOLLM_CUDA_OUTPUT_POOL_MB": "2048",
    "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS": "1",
}


class SelectorTests(unittest.TestCase):
    def run_case(self, changes=None, args=None):
        environment = os.environ.copy()
        for name in ("MOLLM_TEST_CUDA_PREFILL_LIVENESS",
                     "MOLLM_TEST_CUDA_PREFILL_LIVENESS_MAX_SEQ",
                     "MOLLM_TEST_CUDA_PREFILL_STATEFUL_MAX_SEQ"):
            environment.pop(name, None)
        environment.update(OPTIONS)
        for name, value in (changes or {}).items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        result = subprocess.run(args or BASE, env=environment, capture_output=True,
                                text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)  # Missing package even if guard accepts.
        return result.stderr

    def test_invalid_environment_rejected_before_backend(self):
        cases = {
            "MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE": ("", "2", "true", " 1"),
            "MOLLM_RESIDENT_EXPERT_ARENA": (None, "managed"),
            "MOLLM_CUDA_HOST_HEADROOM_MB": (None, "", "0", "24575", "-1", "+24576", " 24576", "24576x", "184467440737095516160"),
            "MOLLM_CUDA_OUTPUT_POOL_MB": (None, "", "0", "4097", "-1", "2048x", "184467440737095516160"),
            "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS": (None,),
            "MOLLM_TEST_CUDA_PREFILL_STATEFUL_MAX_SEQ": ("", "256", "1025", "2048", "+1024", " 1024"),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    error = self.run_case({name: value})
                    if name == "MOLLM_CUDA_HOST_HEADROOM_MB" and value not in (None, "0", "24575"):
                        self.assertIn("CUDA host watchdog: invalid MOLLM_CUDA_HOST_HEADROOM_MB", error)
                    else:
                        self.assertIn("Engine: test-only stateful last-use", error)
                    self.assertNotIn("CudaBackend:", error)

    def test_unsupported_chunk_rejected_before_backend(self):
        for chunk in ("128", "1024"):
            with self.subTest(chunk=chunk):
                args = BASE.copy()
                args[args.index("--prefill-chunk-size") + 1] = chunk
                error = self.run_case(args=args)
                self.assertIn("Engine: test-only stateful last-use requires", error)
                self.assertNotIn("CudaBackend:", error)

    def test_mtp_padding_and_missing_public_reclaim_rejected(self):
        variants = [BASE + ["--mtp-draft-tokens", "1"],
                    BASE + ["--static-padded"],
                    [arg for arg in BASE if arg != "--cuda-prefill-reclaim"]]
        for args in variants:
            with self.subTest(args=args):
                error = self.run_case(args=args)
                self.assertIn("Engine: test-only stateful last-use requires", error)
                self.assertNotIn("CudaBackend:", error)

    def test_valid_selector_reaches_bounded_prefill_setup(self):
        error = self.run_case()
        if "CUDA backend unavailable" in error or "built without MOLLM_CUDA" in error:
            self.skipTest("CUDA backend unavailable")
        self.assertIn("TEST-ONLY prefill stateful read-only last-use contracts enabled", error)
        self.assertIn("sequence cap512", error)
        self.assertNotIn("MoE: loading", error)

    def logical_args(self):
        args = BASE + ["--cuda-prefill-logical-batch"]
        args[args.index("--n-ctx") + 1] = "2048"
        args[args.index("--prefill-chunk-size") + 1] = "2048"
        return args

    def test_logical_batch_early_guards(self):
        args = self.logical_args()
        for name, value, message in (
            ("MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE", "0", "cannot disable"),
            ("MOLLM_TEST_CUDA_PREFILL_STATEFUL_MAX_SEQ", "1024", "owns its2048 cap"),
            ("MOLLM_TEST_CUDA_PREFILL_WINDOW_SCHEDULE", "1", "incompatible with window"),
        ):
            error = self.run_case({name: value}, args)
            self.assertIn(message, error)
            self.assertNotIn("CudaBackend:", error)
        for option, value, message in (
            ("--n-ctx", "0", "invalid value for --n-ctx"),
            ("--prefill-chunk-size", "2049", "--cuda-prefill-logical-batch requires"),
            ("--device", "cpu", "--cuda-prefill-logical-batch requires"),
        ):
            bad = args.copy()
            bad[bad.index(option) + 1] = value
            error = self.run_case(args=bad)
            self.assertIn(message, error)
        error = self.run_case(args=args + ["--mtp-draft-tokens", "1"])
        self.assertIn("--cuda-prefill-logical-batch requires", error)

    def test_logical_batch_reaches_bounded_setup(self):
        error = self.run_case({"MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE": None}, self.logical_args())
        if "CUDA backend unavailable" in error or "built without MOLLM_CUDA" in error:
            self.skipTest("CUDA backend unavailable")
        self.assertIn("sequence cap2048", error)
        self.assertIn("TEST-ONLY prefill stateful read-only", error)
        self.assertNotIn("MoE: loading", error)

    def test_logical_context_and_chunk_are_independent(self):
        # These guards run before opening the nonexistent package. No model
        # allocation or inference is needed to verify the public contract.
        # A chunk is an upper bound, not a minimum prompt/context length.
        # Include short contexts and non-power-of-two/tail-sized chunks.
        for context in ("1", "1024", "2049", "16384", "32768"):
            for chunk in ("0", "1", "128", "957", "1024", "2047", "2048"):
                with self.subTest(context=context, chunk=chunk):
                    args = self.logical_args()
                    args[args.index("--n-ctx") + 1] = context
                    args[args.index("--prefill-chunk-size") + 1] = chunk
                    error = self.run_case(
                        {"MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE": None}, args)
                    if "CUDA backend unavailable" in error or "built without MOLLM_CUDA" in error:
                        self.skipTest("CUDA backend unavailable")
                    self.assertNotIn("--cuda-prefill-logical-batch requires", error)
                    self.assertIn("sequence cap2048", error)
                    self.assertNotIn("MoE: loading", error)

    def test_explicit_1024_cap(self):
        args = BASE.copy()
        args[args.index("--prefill-chunk-size") + 1] = "1024"
        error = self.run_case({"MOLLM_TEST_CUDA_PREFILL_STATEFUL_MAX_SEQ": "1024"}, args)
        if "CUDA backend unavailable" in error or "built without MOLLM_CUDA" in error:
            self.skipTest("CUDA backend unavailable")
        self.assertIn("sequence cap1024", error)
        self.assertIn("TEST-ONLY prefill stateful read-only", error)
        self.assertNotIn("MoE: loading", error)
        error = self.run_case({"MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE": "0",
                               "MOLLM_TEST_CUDA_PREFILL_STATEFUL_MAX_SEQ": "1024"})
        self.assertIn("requires selector1", error)
        self.assertNotIn("CudaBackend:", error)


if __name__ == "__main__":
    unittest.main()
