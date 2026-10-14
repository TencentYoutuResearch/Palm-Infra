import importlib.util
import io
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location("snapshot", Path(__file__).resolve().parents[1]/"tools/check_qsa_indexer_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class HeaderTest(unittest.TestCase):
    def header(self, **changes):
        values = dict(q=2048, past=2048, heads=4, dim=128, rope=64, ratio=4,
                      budget=2048, offset=0, stride=1024, slots=2051, eps=1e-6)
        values.update(changes)
        return b"QSAIDX1\0"+struct.pack("<10If", *values.values())

    def test_real_geometry(self):
        parsed = snapshot.read_header(io.BytesIO(self.header()))
        self.assertEqual((parsed["keys"], parsed["groups"]), (4096, 1024))
        self.assertLess(parsed["elements"]*4, 64*1024*1024)

    def test_invalid_header(self):
        for payload in (b"", b"QSAIDX2\0"+bytes(44), self.header()[:-1]):
            with self.subTest(payload=payload[:8]), self.assertRaises(ValueError):
                snapshot.read_header(io.BytesIO(payload))

    def test_invalid_geometry(self):
        for changes in (dict(q=0), dict(q=2049), dict(past=8192), dict(heads=17),
                        dict(dim=1), dict(rope=63), dict(ratio=0), dict(budget=2049),
                        dict(slots=2050), dict(stride=1023), dict(eps=float("nan")),
                        dict(eps=-1), dict(offset=262145)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                snapshot.read_header(io.BytesIO(self.header(**changes)))

    def test_payload_bound(self):
        with self.assertRaisesRegex(ValueError, "64MiB"):
            snapshot.read_header(io.BytesIO(self.header(heads=16, dim=256, past=6144,
                                                       stride=2048, budget=4080, slots=4083)))


if __name__ == "__main__":
    unittest.main()
