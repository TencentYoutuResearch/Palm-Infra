"""CPU PyTorch checker rejection tests; run in an environment with torch."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location("snapshot", Path(__file__).resolve().parents[1]/"tools/check_qsa_indexer_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
torch.set_num_threads(2)


def fixture():
    generator = torch.Generator().manual_seed(76281)
    data = dict(q=29, past=0, heads=2, dim=8, rope=4, ratio=4, budget=8,
                offset=37, eps=1e-6, keys=29, groups=7)
    data["projection"] = torch.randn(29, 24, generator=generator)
    data["qw"] = torch.randn(8, generator=generator)*.1
    data["kw"] = torch.randn(8, generator=generator)*.1
    data["frequency"] = torch.tensor([1., 0.00031622776])
    data["raw"] = data["projection"][:, 16:].clone()
    data["query"], data["pooled"], data["scores"], visible = snapshot.oracle(data, torch, torch.float32)
    data["ids"] = torch.full((29, 11), -1, dtype=torch.int32)
    for row in range(29):
        groups = (row+1)//4
        chosen = torch.argsort(data["scores"][row, :groups], descending=True, stable=True)[:min(2, groups)]
        ids = sorted([int(block)*4+i for block in chosen for i in range(4)]+list(range(groups*4, row+1)))
        data["ids"][row, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return data


class OracleTest(unittest.TestCase):
    def test_valid(self):
        report = snapshot.check(fixture(), torch)
        self.assertTrue(report["fp32_oracle_pass"])
        self.assertEqual(report["fp32_exact_selected_rows"], 29)
        self.assertFalse(report["whole_model_bf16_certified"])

    def test_corrupt_values(self):
        for name in ("projection", "query", "pooled", "raw", "scores"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                data = fixture()
                if name == "scores":
                    data[name][-1, 0] += 1.
                else:
                    data[name].view(-1)[0] += 1.
                snapshot.check(data, torch)

    def test_future_scores(self):
        for value in (0., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data = fixture()
                data["scores"][0, -1] = value
                snapshot.check(data, torch)

    def test_invalid_selected_ids(self):
        for row, column, value in ((0, 0, -1), (0, 0, 1), (28, 1, 0), (28, 10, -2), (28, 8, -1)):
            with self.subTest(row=row, column=column, value=value), self.assertRaises(ValueError):
                data = fixture()
                data["ids"][row, column] = value
                snapshot.check(data, torch)

    def test_wrong_cutoff(self):
        data = fixture()
        worst = torch.argsort(data["scores"][-1], stable=True)[:2]
        data["ids"][-1] = torch.tensor(sorted([int(block)*4+i for block in worst for i in range(4)])+[28, -1, -1])
        with self.assertRaisesRegex(ValueError, "cutoff"):
            snapshot.check(data, torch)

    def test_truncated_and_trailing_payload(self):
        # Payload itself need not be a semantic fixture: parsing must fail
        # before using any values when a declared tensor is missing or extra.
        import struct
        header = b"QSAIDX1\0"+struct.pack("<10If", 29, 0, 2, 8, 4, 4, 8, 37, 7, 11, 1e-6)
        import io
        size = snapshot.read_header(io.BytesIO(header))["elements"]*4
        with tempfile.TemporaryDirectory(prefix="mollm_qsa_oracle_") as directory:
            path = Path(directory)/"bad.bin"
            for payload in (header+bytes(size-1), header+bytes(size)+b"x"):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    snapshot.read_snapshot(path, torch)


if __name__ == "__main__":
    unittest.main()
