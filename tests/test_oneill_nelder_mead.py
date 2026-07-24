import unittest

import numpy as np

from optimizer.oneill_nelder_mead import minimize_oneill


def rosenbrock(x: np.ndarray) -> float:
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def powell(x: np.ndarray) -> float:
    return float(
        (x[0] + 10.0 * x[1]) ** 2
        + 5.0 * (x[2] - x[3]) ** 2
        + (x[1] - 2.0 * x[2]) ** 4
        + 10.0 * (x[0] - x[3]) ** 4
    )


class OneillNelderMeadTests(unittest.TestCase):
    def test_rosenbrock_matches_as47_reference_counts(self) -> None:
        result = minimize_oneill(
            rosenbrock,
            np.asarray([-1.2, 1.0]),
            step=np.ones(2),
            reqmin=1.0e-8,
            konvge=10,
            maxfev=500,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 157)
        self.assertEqual(result.num_restarts, 0)
        self.assertAlmostEqual(result.fun, 1.2390282440273838e-6, places=15)

    def test_powell_performs_as47_factorial_restarts(self) -> None:
        result = minimize_oneill(
            powell,
            np.asarray([3.0, -1.0, 0.0, 1.0]),
            step=np.ones(4),
            reqmin=1.0e-8,
            konvge=10,
            maxfev=500,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 281)
        self.assertEqual(result.num_restarts, 4)
        self.assertAlmostEqual(result.fun, 6.473222382959405e-6, places=15)


if __name__ == "__main__":
    unittest.main()
