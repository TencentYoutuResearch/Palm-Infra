"""CPU algebra proof only: not a CUDA equivalence or model-quality gate."""
import sys
import unittest
from pathlib import Path
try:
    import numpy as np
except ImportError:
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_gdn_chunkwise import chunkwise, sequential, coefficients, fixture


class ChunkwiseAlgebraTest(unittest.TestCase):
    def test_f64_scalar_equivalence_shapes_and_tails(self):
        for n, kd, vd in ((1, 3, 5), (17, 7, 9), (65, 128, 128), (129, 8, 13)):
            for chunk in (1, 16, 32, 64, 128):
                for variant in range(4):
                    with self.subTest(n=n, chunk=chunk, variant=variant):
                        args = fixture(n, kd, vd, variant, np.float64)
                        before = [x.copy() for x in args]
                        expected, state = sequential(*args)
                        actual, final = chunkwise(*args, chunk=chunk)
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
                        np.testing.assert_allclose(final, state, rtol=0, atol=1e-11)
                        for a, b in zip(args, before):
                            np.testing.assert_array_equal(a, b)

    def test_f32_long_prompt_output_and_final_state(self):
        for variant in range(4):
            args = fixture(1743, 128, 128, variant, np.float32)
            expected, state = sequential(*args)
            for chunk in (16, 32, 64, 128):
                actual, final = chunkwise(*args, chunk=chunk)
                self.assertTrue(np.isfinite(actual).all() and np.isfinite(final).all())
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
                np.testing.assert_allclose(final, state, rtol=0, atol=1e-5)

    def test_zero_decay_and_underflow_do_not_divide_prefixes(self):
        for variant in (1, 2):
            args = fixture(65, 128, 128, variant, np.float32)
            self.assertEqual(np.cumprod(args[3])[-1], 0)
            with np.errstate(divide='raise', invalid='raise', over='raise'):
                out, state = chunkwise(*args, chunk=64)
            self.assertTrue(np.isfinite(out).all() and np.isfinite(state).all())

    def test_cross_window_carry_matches_whole_prompt(self):
        for dtype in (np.float64, np.float32):
            args = fixture(1743, 128, 128, 0, dtype)
            q, k, v, d, b, initial = args
            expected, state = sequential(*args)
            out_a, carry = chunkwise(q[:1024], k[:1024], v[:1024], d[:1024], b[:1024], initial)
            out_b, final = chunkwise(q[1024:], k[1024:], v[1024:], d[1024:], b[1024:], carry)
            tolerance = 1e-11 if dtype == np.float64 else 1e-5
            np.testing.assert_allclose(np.concatenate((out_a, out_b)), expected, rtol=0, atol=tolerance)
            np.testing.assert_allclose(final, state, rtol=0, atol=tolerance)

    def test_coefficients_are_independent_of_initial_state(self):
        q, k, v, d, b, state = fixture(17, 7, 9, 1, np.float64)
        u, w, qp, attention, tail, end_decay = coefficients(q, k, v, d, b)
        for initial in (state, state * -3, np.zeros_like(state)):
            delta = u - w @ initial
            output = (qp @ initial + attention @ delta) * .125
            final = end_decay * initial + tail.T @ delta
            expected, expected_state = sequential(q, k, v, d, b, initial)
            np.testing.assert_allclose(output, expected, rtol=0, atol=1e-11)
            np.testing.assert_allclose(final, expected_state, rtol=0, atol=1e-11)

    def test_repeated_key_head_has_independent_value_head_gates_and_state(self):
        # Actual package has sixteen key heads, forty-eight value heads.
        # Test the same repeat=3 mapping with two independent key groups.
        for key_head in range(2):
            q, k, _, _, _, _ = fixture(65, 128, 128, key_head, np.float64)
            for repeat in range(3):
                _, _, v, decay, beta, state = fixture(65, 128, 128, repeat, np.float64)
                expected, final = sequential(q, k, v, decay, beta, state)
                actual, candidate = chunkwise(q, k, v, decay, beta, state, chunk=32)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
                np.testing.assert_allclose(candidate, final, rtol=0, atol=1e-11)

    def test_invalid_dimensions_values_precision_and_chunk(self):
        args = fixture(17, 7, 9, 0, np.float32)
        for chunk in (0, -1, 129, 1.5, True):
            with self.assertRaises(ValueError):
                chunkwise(*args, chunk=chunk)
        for index, replacement in ((0, args[0].ravel()), (1, args[1][:, :-1]),
                                   (2, args[2][:-1]), (3, args[3][:, None]),
                                   (5, args[5][:-1]), (0, args[0].astype(np.float64))):
            bad = list(args); bad[index] = replacement
            with self.assertRaises(ValueError):
                chunkwise(*bad)
        for index, value in ((0, np.nan), (1, np.inf), (3, -1), (3, 2), (4, -1), (4, 2)):
            bad = [x.copy() for x in args]; bad[index].flat[0] = value
            with self.assertRaises(ValueError):
                chunkwise(*bad)


if __name__ == '__main__':
    unittest.main()
