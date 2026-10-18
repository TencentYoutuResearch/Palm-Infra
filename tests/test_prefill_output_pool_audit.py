import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_prefill_outputs import simulate_output_pool, gr_inject_consumers, stateful_protected_ids, dense_projection_shapes
from audit_prefill_outputs import leading_split_strides, declared_layout_copies


class OutputPoolAuditTest(unittest.TestCase):
    def test_leading_split_preserves_every_address_in_padded_rows(self):
        for rows in (1, 7, 17):
            for width, heads in ((6, 3), (512, 2), (6144, 24)):
                source = (width, rows, 1, 1)
                strides = (4, (width + 19) * 4, (width + 19) * rows * 4,
                           (width + 19) * rows * 4)
                target = (width // heads, heads, rows, 1)
                result = leading_split_strides(source, strides, target, 4)
                self.assertIsNotNone(result)
                for s in range(rows):
                    for h in range(heads):
                        for d in range(target[0]):
                            self.assertEqual(d * result[0] + h * result[1] + s * result[2],
                                             (d + h * target[0]) * 4 + s * strides[1])

    def test_leading_split_does_not_merge_permuted_or_invalid_layout(self):
        self.assertIsNone(leading_split_strides((2, 7, 3, 1), (4, 100, 8, 700),
                                                (2, 21, 1, 1), 4))
        for shape, strides, target, es in [
                ((6, 7, 1, 1), (8, 100, 700, 700), (2, 3, 7, 1), 4),
                ((6, 7, 1, 1), (4, 100, 700, 700), (2, 4, 7, 1), 4),
                ((6, 7, 1, 1), (4, 100, 700, 700), (2, 3, 8, 1), 4),
                ((6, 7, 1, 1), (4, 0, 700, 700), (2, 3, 7, 1), 4),
                ((6, 7, 1, 1), (4, 100, 700, 700), (0, 3, 7, 1), 4),
                ((6, 7, 1, 1), (4, 100, 700, 700), (2, 3, 7, 1), 0)]:
            self.assertIsNone(leading_split_strides(shape, strides, target, es))

    @staticmethod
    def layout_node(index, op, shape, inputs=(), integers=(), precision=0):
        return dict(id=index, op=op, shape=shape, inputs=list(inputs),
                    integers=list(integers), precision=precision, dim_expr=[(0, 0)] * 4)

    def test_declared_copy_audit_keeps_current_layout_not_proposal(self):
        node = self.layout_node
        nodes = [node(0, 10, (10, 7, 1, 1)),
                 node(1, 63, (6, 7, 1, 1), [0], [0, 2, 6]),
                 node(2, 60, (2, 3, 7, 1), [1]),
                 node(3, 61, (2, 7, 3, 1), [2], [0, 2, 1, 3]),
                 node(4, 60, (2, 21, 1, 1), [3])]
        records = declared_layout_copies(nodes, 7)
        self.assertEqual([r['id'] for r in records], [2, 4])
        self.assertEqual([r['leading_split_candidate'] for r in records], [True, False])
        self.assertEqual([r['bytes'] for r in records], [168, 168])
        self.assertEqual(records[0]['consumers'], [3])
        nodes[1]['precision'] = 7
        self.assertEqual(declared_layout_copies(nodes, 7), [])

    def test_declared_copy_audit_contiguous_alias_and_malformed_fail_closed(self):
        node = self.layout_node
        nodes = [node(0, 10, (6, 7, 1, 1)), node(1, 60, (2, 3, 7, 1), [0])]
        self.assertEqual(declared_layout_copies(nodes, 7), [])
        nodes[1]['shape'] = (2, 4, 7, 1)
        with self.assertRaises(ValueError):
            declared_layout_copies(nodes, 7)
        nodes[1]['inputs'] = []
        with self.assertRaises(ValueError):
            declared_layout_copies(nodes, 7)
        with self.assertRaises(ValueError):
            declared_layout_copies(nodes[:1], 0)

    def test_declared_copy_audit_dynamic_rows_and_contiguous_are_distinct(self):
        node = self.layout_node
        nodes = [node(0, 10, (10, 7, 1, 1)),
                 node(1, 63, (6, 7, 1, 1), [0], [0, 2, 6]),
                 node(2, 65, (6, 7, 1, 1), [1])]
        for n in nodes:
            n['dim_expr'][1] = (1, 0)
        self.assertEqual(declared_layout_copies(nodes, 1), [])
        record, = declared_layout_copies(nodes, 17)
        self.assertEqual(record['bytes'], 6 * 17 * 4)
        self.assertFalse(record['leading_split_candidate'])
        nodes[2]['precision'] = 1
        self.assertEqual(declared_layout_copies(nodes, 17), [])

    def test_declared_dense_shapes_keep_moe_separate(self):
        weights = [dict(id=i, op=1, inputs=[], shape=shape, dim_expr=[(0, 0)] * 4)
                   for i, shape in enumerate([(2560, 6144, 1, 1), (6144, 2560, 1, 1)])]
        nodes = weights + [dict(id=2, op=10, inputs=[0, 0]),
                           dict(id=3, op=120, inputs=[0, 1, 0, 0, 1, 1, 0, 0])]
        counts, unresolved = dense_projection_shapes(nodes)
        self.assertEqual(dict(counts), {('matmul', 2560, 6144): 1,
                                      ('moe_router_shared', 6144, 2560): 3,
                                      ('moe_router_shared', 2560, 6144): 2})
        self.assertEqual(unresolved, 0)

    def test_declared_dense_shapes_do_not_guess_dynamic_or_views(self):
        constant = dict(id=0, op=1, inputs=[], shape=(2560, 6144, 1, 1), dim_expr=[(0, 0)] * 4)
        for change in [dict(op=60), dict(shape=(2560, 6144, 2, 1)),
                       dict(dim_expr=[(1, 0)] * 4), dict(shape=(-1, 6144, 1, 1))]:
            nodes = [{**constant, **change}, dict(id=1, op=10, inputs=[0, 0])]
            counts, unresolved = dense_projection_shapes(nodes)
            self.assertFalse(counts)
            self.assertEqual(unresolved, 1)
        self.assertEqual(dense_projection_shapes([dict(id=0, op=10, inputs=[])]), ({}, 1))

    def test_recurrent_contract_keeps_only_mutable_state(self):
        for op, count, mutable in [(140, 3, 2), (111, 8, 7)]:
            node = dict(id=20, op=op, precision=0, inputs=list(range(count)))
            self.assertEqual(stateful_protected_ids(node), [20, *range(count)])
            self.assertEqual(stateful_protected_ids(node, True), [mutable])

    def test_contract_unsupported_shape_precision_or_op_fails_closed(self):
        for op, precision, inputs in [(140, 0, [1, 2]), (111, 0, [1, 2, 3]),
                                      (140, 1, [1, 2, 3]), (161, 0, [1, 2, 3])]:
            node = dict(id=20, op=op, precision=precision, inputs=inputs)
            self.assertEqual(stateful_protected_ids(node, True), [20, *inputs])
        node = dict(id=20, op=140, inputs=[1, 2, 3])
        self.assertEqual(stateful_protected_ids(node, True), [20, 1, 2, 3])

    def test_nonstateful_has_no_contract_protection(self):
        node = dict(id=20, op=10, precision=0, inputs=[1, 2])
        self.assertEqual(stateful_protected_ids(node, True), [])

    def test_cuda_contracts_match_native_mutable_inputs(self):
        for op, count, protected in [(24, 6, []), (50, 6, [4, 5]),
                                     (50, 7, [4, 5]), (140, 3, [2]), (111, 8, [7])]:
            node = dict(id=20, op=op, precision=0, inputs=list(range(count)))
            self.assertEqual(stateful_protected_ids(node, cuda_contracts=True), protected)
            # Older modes intentionally keep the existing broader scenario.
            if op in (24, 50):
                self.assertEqual(stateful_protected_ids(node, True), [20, *range(count)])

    def test_cuda_contracts_fail_closed_and_keep_exact_input_ids(self):
        for op, count in [(24, 5), (24, 7), (50, 5), (50, 8), (51, 6)]:
            node = dict(id=20, op=op, precision=0, inputs=list(range(count)))
            self.assertEqual(stateful_protected_ids(node, cuda_contracts=True), [20, *range(count)])
        for precision in (1, 2, None):
            node = dict(id=20, op=50, precision=precision, inputs=[2, 3, 7, 8, 11, 13])
            self.assertEqual(stateful_protected_ids(node, cuda_contracts=True), [20, 2, 3, 7, 8, 11, 13])
        node = dict(id=20, op=50, precision=0, inputs=[2, 3, 7, 8, 11, 13, 17])
        self.assertEqual(stateful_protected_ids(node, cuda_contracts=True), [11, 13])

    def test_inject_consumers_require_single_rms_nonoutput(self):
        nodes = [dict(id=0, op=134, inputs=[]), dict(id=1, op=25, inputs=[0]),
                 dict(id=2, op=134, inputs=[0]), dict(id=3, op=25, inputs=[2])]
        records = gr_inject_consumers(nodes, [])
        self.assertEqual(records[0]['ops'], [25, 134])
        self.assertFalse(records[0]['removable'])
        self.assertTrue(records[1]['removable'])
        self.assertFalse(gr_inject_consumers(nodes, [2])[1]['removable'])

    def test_inject_duplicate_edge_and_non_rms_are_not_eligible(self):
        nodes = [dict(id=0, op=134, inputs=[]), dict(id=1, op=25, inputs=[0, 0])]
        self.assertFalse(gr_inject_consumers(nodes, [])[0]['removable'])
        nodes[1]['inputs'] = [0]
        nodes[1]['op'] = 10
        self.assertFalse(gr_inject_consumers(nodes, [])[0]['removable'])

    def test_best_fit_reuses_original_allocation_size(self):
        reserved, peak, free = simulate_output_pool([12, 5, 8], {0: [0], 1: [1]})
        self.assertEqual((reserved, peak, free), (12, 12, [12]))

    def test_overlap_and_best_fit(self):
        reserved, peak, free = simulate_output_pool([12, 8, 7], {1: [0, 1]})
        self.assertEqual((reserved, peak, free), (20, 20, [8, 12]))

    def test_changed_shape_keeps_pool_reservations(self):
        reserved, peak, free = simulate_output_pool([12, 8], {})
        self.assertEqual((reserved, peak, free), (20, 20, [8, 12]))
        self.assertEqual(simulate_output_pool([6, 10], {}, free), (20, 20, [8, 12]))
        self.assertEqual(simulate_output_pool([13], {}, free), (33, 13, [8, 12, 13]))

    def test_nonowning_views(self):
        self.assertEqual(simulate_output_pool([0, 8, 0], {2: [0, 2]}), (8, 8, [8]))

    def test_invalid_sizes_and_double_release(self):
        for sizes, releases, free in [([-1], {}, []), ([1], {}, [-1]),
                                      ([1, 1], {0: [0], 1: [0]}, [])]:
            with self.assertRaises(ValueError):
                simulate_output_pool(sizes, releases, free)


if __name__ == '__main__':
    unittest.main()
