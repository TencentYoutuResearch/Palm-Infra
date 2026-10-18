import copy
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibration_audit", ROOT / "tools/audit_nvfp4_input_calibration.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class CalibrationAuditTest(unittest.TestCase):
    def fixture(self, directory, values=(1., 1., 2., 3.), modify=None):
        data = struct.pack("<4f", *values)
        weight = audit_module.WEIGHT.pack(0x50414D58, 0, 2, 0, 2, 2, 0, 0,
                                         88, len(data), 0, 0, 0, 0) + data
        calibration = {"weight": "scale", "weight_offset": 0,
                       "data_bytes": 16, "data_offset": 88, "scales_per_expert": 2}
        metadata = {"weights": {"scale": [0, len(weight)]},
                    "moe_expert_storage": {"num_experts": 2, "layers": [
                        {"layer": 0, "gate_up": {"input_scales": calibration}}]}}
        if modify:
            modify(metadata)
        raw_meta = json.dumps(metadata).encode()
        weight_offset = 128 + len(raw_meta)
        header = audit_module.PACKAGE.pack(0x4D4C4F4D, 1, 128, len(raw_meta),
                                          0, 0, 0, 0, 0, 0, 0, 0,
                                          weight_offset, len(weight), 0, 0, 0)
        path = pathlib.Path(directory) / "test.mollm"
        path.write_bytes(header + raw_meta + weight)
        return path

    def test_exact_pairs_and_unique_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_module.audit(self.fixture(directory))
        self.assertEqual(result["expert_pairs"], 2)
        self.assertEqual(result["equal_gate_up_pairs"], 1)
        self.assertEqual(result["layers"][0]["unique_both_globals"], 3)
        self.assertEqual(result["calibration_payload_bytes_read"], 16)

    def test_uniform(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_module.audit(self.fixture(directory, (1.,) * 4))
        self.assertEqual(result["equal_gate_up_pairs"], 2)
        self.assertEqual(result["layers"][0]["unique_gate_up_pairs"], 1)

    def test_reject_invalid_scales(self):
        for value in (0., -1., float("inf"), float("nan")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = self.fixture(directory, (value, 1., 1., 1.))
                with self.assertRaises(ValueError):
                    audit_module.audit(path)

    def test_reject_metadata_mismatch(self):
        for key, value in (("weight_offset", 1), ("data_offset", 87),
                           ("data_bytes", 12), ("scales_per_expert", 1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = self.fixture(directory, modify=lambda m: m["moe_expert_storage"]["layers"][0]["gate_up"]["input_scales"].update({key: value}))
                with self.assertRaises(ValueError):
                    audit_module.audit(path)

    def test_reject_duplicate_layers(self):
        def duplicate(metadata):
            layers = metadata["moe_expert_storage"]["layers"]
            layers.append(copy.deepcopy(layers[0]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                audit_module.audit(self.fixture(directory, modify=duplicate))

    def test_reject_truncated_package(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                audit_module.audit(path)


if __name__ == "__main__":
    unittest.main()
