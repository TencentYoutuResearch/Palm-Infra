"""Offline fixtures for logical projection-payload accounting (not DRAM)."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('traffic', Path(__file__).resolve().parents[1] / 'tools/audit_decode_weight_traffic.py')
traffic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(traffic)


def string(value):
    data = value.encode()
    return struct.pack('<I', len(data)) + data


def vector(values):
    return struct.pack('<I' + 'i' * len(values), len(values), *values)


def node(index, op, inputs=(), integers=(), strings=()):
    return (struct.pack('<iI', index, op) + vector(inputs) + struct.pack('<4q', 1, 1, 1, 1)
            + bytes(32) + struct.pack('<I', 0) + vector(integers) + vector(())
            + struct.pack('<I', len(strings)) + b''.join(string(s) for s in strings))


def weight(precision, flags, rows, cols, data, scales=0):
    return (struct.pack('<4I4Q4Q2I', 0x50414D58, flags, 2, precision, rows, cols, 1, 1,
                        88, data, 88 + data if scales else 0, scales, 32, cols // 32)
            + bytes(data + scales))


class TrafficTest(unittest.TestCase):
    def fixture(self, routed=False, head=True):
        blobs = {'dense': weight(3, 4, 4, 32, 160)}
        nodes = [node(0, 0), node(1, 1, strings=('dense',)), node(2, 10, (0, 1))]
        layers = []
        if routed:
            for name in ('gate', 'down'):
                blobs[name] = weight(8, 16, 4, 32, 64, 24)
            nodes += [node(3, 1, strings=('gate',)), node(4, 1, strings=('down',)),
                      node(5, 120, (0, 1, 3, 4, 1, 1, 1, 1), (32, 2, 2))]
            layers = [dict(gate_up=dict(weight='gate', expert_data_bytes=32, expert_scales_bytes=12),
                           down=dict(weight='down', expert_data_bytes=32, expert_scales_bytes=12))]
        if head:
            blobs['./lm_head.weights'] = weight(2, 0, 2, 32, 64, 8)
        graph = struct.pack('<4I', 0x4D4C4C47, 3, len(nodes), 0) + vector(()) + vector(()) + b''.join(nodes)
        spans, payload = {}, b''
        for name, blob in blobs.items():
            spans[name] = [len(payload), len(blob)]
            payload += blob
        metadata = json.dumps(dict(weights=spans, moe_expert_storage=dict(layers=layers))).encode()
        fields = [128, len(metadata), 0, 0, 0, 0, 0, 0, 128 + len(metadata), len(graph),
                  128 + len(metadata) + len(graph), len(payload), 0, 0, 0]
        data = struct.pack('<II15Q', 0x4D4C4F4D, 1, *fields) + metadata + graph + payload
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.mollm'
            path.write_bytes(data)
            return traffic.audit(path)

    def test_padding_and_external_head(self):
        result = self.fixture()
        self.assertEqual(result['categories']['dense_matmul']['bytes'], 80)
        self.assertEqual(result['categories']['dense_matmul']['associated_storage_bytes'], 160)
        self.assertEqual(result['categories']['lm_head']['bytes'], 72)
        self.assertEqual(result['logical_projection_bytes'], 152)

    def test_top_k_not_full_expert_array(self):
        result = self.fixture(routed=True)
        self.assertEqual(result['categories']['moe_routed']['bytes'], 176)
        self.assertEqual(result['categories']['moe_router_shared']['bytes'], 400)
        self.assertEqual(result['logical_projection_bytes'], 728)

    def test_missing_head_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lm_head'):
            self.fixture(head=False)

    def test_truncated_graph(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            traffic.graph_nodes(b'')

    def test_symbolic_dimensions_preserved(self):
        serialized = bytearray(node(0, 0))
        # id/op, empty input vector, four int64 dimensions precede DimExpr.
        struct.pack_into('<b3xi', serialized, 44, 2, 4)
        data = struct.pack('<6I', 0x4D4C4C47, 3, 1, 0, 0, 0) + serialized
        self.assertEqual(traffic.graph_nodes(data)[0]['dim_expr'],
                         [(2, 4), (0, 0), (0, 0), (0, 0)])
        serialized[44] = 6
        with self.assertRaisesRegex(ValueError, 'dimension kind'):
            traffic.graph_nodes(struct.pack('<6I', 0x4D4C4C47, 3, 1, 0, 0, 0)
                                + serialized)

    def test_wrong_graph_version(self):
        with self.assertRaisesRegex(ValueError, 'v3'):
            traffic.graph_nodes(struct.pack('<3I', 0x4D4C4C47, 2, 0))

    def test_huge_vector_count_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, 'vector'):
            traffic.Reader(struct.pack('<I', 0xffffffff)).vector('i')

    def test_forward_node_reference_rejected(self):
        data = struct.pack('<6I', 0x4D4C4C47, 3, 1, 0, 0, 0) + node(0, 10, (1, 1))
        with self.assertRaisesRegex(ValueError, 'preceding'):
            traffic.graph_nodes(data)

    def test_trailing_graph_data(self):
        with self.assertRaisesRegex(ValueError, 'trailing'):
            traffic.graph_nodes(struct.pack('<6I', 0x4D4C4C47, 3, 0, 0, 0, 0) + b'x')


if __name__ == '__main__':
    unittest.main()
