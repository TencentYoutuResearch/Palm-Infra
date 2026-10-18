"""CLI help and parse failures; never initialize a backend or open a model."""
import subprocess
import sys
import unittest

BINARIES = sys.argv[1:]
sys.argv[1:] = []


class CliDiagnosticsTests(unittest.TestCase):
    def test_explicit_help(self):
        for binary in BINARIES:
            with self.subTest(binary=binary):
                result = subprocess.run([binary, "--help"], capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 0)
                self.assertIn("Usage:", result.stdout)
                self.assertIn("--cuda-prefill-logical-batch", result.stdout)
                self.assertIn("independent --n-ctx", result.stdout)
                self.assertIn("not a required prompt length", result.stdout)
                self.assertIn("without logical batching: ctx<=4096", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_invalid_arguments_show_error_without_full_help(self):
        for binary in BINARIES:
            for args, message in (([], "missing required --package"),
                                  (["--package", "never-open.mollm", "--n-ctx", "0"],
                                   "invalid value for --n-ctx"),
                                  (["--unknown-option"], "unknown argument")):
                with self.subTest(binary=binary, args=args):
                    result = subprocess.run([binary] + args, capture_output=True,
                                            text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertIn("--help", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertNotIn("CudaBackend:", result.stderr)


if __name__ == "__main__":
    if len(BINARIES) != 2:
        raise SystemExit("requires chat and bench binary paths")
    unittest.main()
