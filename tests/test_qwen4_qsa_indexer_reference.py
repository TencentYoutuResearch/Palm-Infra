"""QSA CPU computation/state checks; not a CUDA/model-quality gate."""
import sys
import unittest
from pathlib import Path
try:
    import numpy as np
except ImportError:
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from qwen4_qsa_indexer_reference import QsaIndexerReference, centered_rms_norm, partial_rope


def fixture(dtype=np.float32):
    rng = np.random.default_rng(1817)
    weights = rng.normal(size=(24, 7)).astype(dtype)
    qnorm = rng.normal(scale=.1, size=8).astype(dtype)
    knorm = rng.normal(scale=.1, size=8).astype(dtype)
    hidden = rng.normal(size=(73, 7)).astype(dtype)
    def make():
        return QsaIndexerReference(weights, qnorm, knorm, heads=2, budget=8, rope_dim=4)
    return make, hidden


class QsaIndexerTests(unittest.TestCase):
    def test_centered_weight_and_shared_half_split_rope(self):
        x = np.array([[3., 4.]], dtype=np.float32)
        expected = np.array([[3., 8.]]) / np.sqrt(12.5 + 1e-6)
        np.testing.assert_allclose(centered_rms_norm(x, np.array([0., 1.]), 1e-6), expected, rtol=1e-6)
        x = np.arange(1, 9, dtype=np.float32)[None]
        actual = partial_rope(x, [3], 4, 10000.)
        angles = np.array([3., .03])
        expected = x.copy()
        expected[0, :2] = x[0, :2] * np.cos(angles) - x[0, 2:4] * np.sin(angles)
        expected[0, 2:4] = x[0, 2:4] * np.cos(angles) + x[0, :2] * np.sin(angles)
        np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=1e-6)
        np.testing.assert_array_equal(actual[:, 4:], x[:, 4:])

    def test_partial_groups_chunk_decode_and_reset(self):
        for dtype in (np.float16, np.float32):
            make, hidden = fixture(dtype)
            whole = make()
            expected = whole.append(hidden)
            for chunks in ([1] * 73, [3, 2, 1, 5, 7, 19, 36], [32, 32, 9]):
                cached = make()
                outputs, offset = [], 0
                for chunk in chunks:
                    outputs.extend(cached.append(hidden[offset:offset + chunk]))
                    offset += chunk
                self.assertEqual(outputs, expected)
                np.testing.assert_array_equal(cached.raw_keys, whole.raw_keys)
                np.testing.assert_array_equal(cached.positions, whole.positions)
                cached.reset()
                self.assertIsNone(cached.raw_keys)
                self.assertEqual(cached.append(hidden), expected)
            for row in range(11):
                self.assertEqual(expected[row], list(range(row + 1)))
            self.assertEqual(len(expected[11]), 8)
            self.assertEqual(len(expected[12]), 9)

    def test_pool_before_norm_and_cache_raw_keys(self):
        # Normalize-before-pool incorrectly cancels the first group's keys.
        indexer = QsaIndexerReference(np.eye(4, dtype=np.float32), np.zeros(2), np.zeros(2),
                                      heads=1, budget=4, rope_dim=2)
        hidden = np.array([[1, 0, 10, 0], [1, 0, -1, 0], [1, 0, 10, 0], [1, 0, -1, 0],
                           [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], dtype=np.float32)
        self.assertEqual(indexer.append(hidden, positions=np.zeros(8, dtype=np.int64))[-1], [0, 1, 2, 3])
        np.testing.assert_array_equal(indexer.raw_keys, hidden[:, 2:])

    def test_mask_gaps_group_visible_tokens_and_retain_tail(self):
        make, hidden = fixture()
        visibility = np.arange(17)[None, :] <= np.arange(17)[:, None]
        visibility[:, [0, 3, 9]] = False
        outputs = make().append(hidden[:17], visible=visibility)
        visible = np.flatnonzero(visibility[-1]).tolist()
        result = outputs[-1]
        self.assertEqual(len(result), 10)
        self.assertEqual(result[-2:], visible[-2:])
        for start in range(0, 12, 4):
            self.assertIn(len(set(visible[start:start + 4]) & set(result)), (0, 4))
        self.assertFalse(set([0, 3, 9]) & set(result))

    def test_invalid_append_is_atomic(self):
        make, hidden = fixture()
        indexer = make()
        indexer.append(hidden[:3])
        before = indexer.raw_keys.copy()
        for kwargs in ({'positions': [-1]}, {'visible': np.ones((1, 5), dtype=bool)},
                       {'positions': [1.5]}):
            with self.assertRaises(ValueError):
                indexer.append(hidden[3:4], **kwargs)
            np.testing.assert_array_equal(indexer.raw_keys, before)
            self.assertEqual(len(indexer.positions), 3)
        with self.assertRaises(ValueError):
            indexer.append(hidden[3:4].astype(np.float16))

    def test_independent_scalar_scores_with_offset_positions_and_mask_gaps(self):
        import math
        make, hidden = fixture()
        indexer = make()
        positions = np.arange(100, 117, dtype=np.int64)
        mask = np.arange(17)[None, :] <= np.arange(17)[:, None]
        mask[:, [0, 3]] = False
        actual = indexer.append(hidden[:17], positions, mask)
        # Separate scalar derivation: no shared normalization/rotation/selection
        # helpers. Nonzero offsets catch rotating a pooled key at block ID,
        # group end, or compacted-visible index rather than first token position.
        projected = hidden[:17].astype(np.float64) @ indexer.projection.astype(np.float64).T
        def normalize_rotate(vector, weight, position):
            norm = math.sqrt(sum(float(x) ** 2 for x in vector) / 8 + 1e-6)
            out = [float(x) / norm * (1 + float(w)) for x, w in zip(vector, weight)]
            result = out.copy()
            for j in range(2):
                angle = position / 10000000. ** (2 * j / 4)
                result[j] = out[j] * math.cos(angle) - out[j + 2] * math.sin(angle)
                result[j + 2] = out[j + 2] * math.cos(angle) + out[j] * math.sin(angle)
            return result
        for row in range(17):
            ids = np.flatnonzero(mask[row]).tolist()
            groups = [ids[start:start + 4] for start in range(0, len(ids) - 3, 4)]
            queries = [normalize_rotate(projected[row, h * 8:(h + 1) * 8],
                                        indexer.q_norm, positions[row]) for h in range(2)]
            scores = []
            for group in groups:
                pooled = [sum(projected[token, 16 + d] for token in group) / 4 for d in range(8)]
                key = normalize_rotate(pooled, indexer.k_norm, positions[group[0]])
                scores.append(sum(max(0., sum(q[d] * key[d] for d in range(8)))
                                  for q in queries) / math.sqrt(8))
            ordered = sorted(range(len(groups)), key=lambda g: (-scores[g], g))
            if len(groups) > 2:
                self.assertGreater(scores[ordered[1]] - scores[ordered[2]], 1e-5)
            selected = [token for group in ordered[:2] for token in groups[group]]
            selected += ids[4 * len(groups):]
            self.assertEqual(actual[row], sorted(selected))


if __name__ == '__main__':
    unittest.main()
