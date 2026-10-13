"""QSA selection semantics; CPU-only, no model payload or GPU initialization."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from qwen4_qsa_reference import select_qsa_tokens


class QsaSelectionTests(unittest.TestCase):
    def test_dense_equivalence_and_first_sparse_transition(self):
        for length in (0, 1, 2048, 2049, 2050, 2051):
            ids = list(range(length))
            self.assertEqual(select_qsa_tokens(ids, list(range(length // 4))), ids)
        self.assertEqual(select_qsa_tokens(range(2052), range(513)), list(range(4, 2052)))

    def test_complete_groups_ranked_but_tail_always_retained(self):
        # A masked gap is not part of a group; groups count visible tokens.
        ids = [0, 2, 3, 8, 11, 12, 14, 17, 19, 20, 21, 25, 26, 30, 31]
        self.assertEqual(select_qsa_tokens(ids, [10., 1., 9.], budget=8),
                         ids[:4] + ids[8:])
        result = select_qsa_tokens(range(32768), range(8192))
        self.assertEqual(result, list(range(30720, 32768)))

    def test_dense_execution_is_not_a_qsa_reference(self):
        chosen = select_qsa_tokens(range(12), [1., 2., 3.], budget=8)
        # Equal attention logits -> uniform probabilities. The discarded block
        # carries nonzero values, so dense and sparse outputs actually differ.
        values = [12.] * 4 + [0.] * 8
        dense = sum(values) / len(values)
        sparse = sum(values[i] for i in chosen) / len(chosen)
        self.assertEqual(dense, 4.)
        self.assertEqual(sparse, 0.)

    def test_tie_policy_is_explicit_and_invalid_inputs_rejected(self):
        self.assertEqual(select_qsa_tokens(range(12), [1., 1., 1.], budget=8), list(range(8)))
        for ids, scores, budget, ratio in (([1, 1], [], 8, 4), ([-1], [], 8, 4),
                ([2, 1], [], 8, 4), (range(4), [], 8, 4),
                (range(4), [float("nan")], 8, 4), (range(4), [float("inf")], 8, 4),
                ([], [], 0, 4), ([], [], 7, 4), ([], [], 8, 0)):
            with self.assertRaises(ValueError):
                select_qsa_tokens(ids, scores, budget, ratio)


if __name__ == "__main__":
    unittest.main()
