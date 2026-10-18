import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools/compare_token_loss.py"
spec = importlib.util.spec_from_file_location("token_loss_compare", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TokenLossCompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mollm_loss_compare_")
        self.addCleanup(self.tmp.cleanup)
        self.control = Path(self.tmp.name) / "control.csv"
        self.candidate = Path(self.tmp.name) / "candidate.csv"

    def write(self, path, rows):
        with path.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["position", "target", "ce"])
            writer.writerows(rows)

    def test_exact_and_roundoff(self):
        self.write(self.control, [(0, 11, 1.), (1, 12, 2.)])
        self.write(self.candidate, [(0, 11, 1.), (1, 12, 2.000002)])
        report = module.compare(self.control, self.candidate)
        self.assertEqual(report["tokens"], 2)
        self.assertEqual(report["exact_tokens"], 1)
        self.assertAlmostEqual(report["mean_abs"], 1e-6)
        self.assertEqual(subprocess.run([sys.executable, str(TOOL), str(self.control), str(self.candidate)],
                                       capture_output=True).returncode, 0)

    def test_lower_loss_is_not_equivalence(self):
        self.write(self.control, [(0, 11, 9.)])
        self.write(self.candidate, [(0, 11, 8.)])
        self.assertEqual(subprocess.run([sys.executable, str(TOOL), str(self.control), str(self.candidate)],
                                       capture_output=True).returncode, 1)

    def test_bad_alignment_and_nonfinite_fail(self):
        self.write(self.control, [(0, 11, 1.)])
        for rows in ([], [(0, 12, 1.)], [(1, 11, 1.)], [(0, 11, "nan")], [(0, 11, "inf")], [(0, 11, -1.)]):
            self.write(self.candidate, rows)
            with self.assertRaises(ValueError):
                module.compare(self.control, self.candidate)


if __name__ == "__main__":
    unittest.main()
