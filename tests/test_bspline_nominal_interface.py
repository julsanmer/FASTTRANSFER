import inspect
import unittest

import numpy as np

from experiments.reproduce_gondelach_fig2 import inclusive_grid
from experiments.run_bspline_variant_analysis import config_run_id
from optimizer.helpers_Bspline import build_bspline_matrices
from optimizer.optimization_Bspline_freetf import (
    solve_free_tf_cartesian_bspline,
    solve_free_tf_cylindrical_bspline,
)


REMOVED_PARAMETERS = {
    "ghost_controls",
    "seed_velocity_weight",
    "objective",
    "winding_weight",
    "winding_constraint",
}


class BsplineNominalInterfaceTests(unittest.TestCase):
    def test_solvers_expose_only_the_nominal_delta_v_formulation(self) -> None:
        for solver in (
            solve_free_tf_cartesian_bspline,
            solve_free_tf_cylindrical_bspline,
        ):
            parameters = set(inspect.signature(solver).parameters)
            self.assertFalse(REMOVED_PARAMETERS & parameters)

    def test_basis_uses_exactly_the_requested_control_points(self) -> None:
        matrices = build_bspline_matrices(n_ctrl=8, degree=3, n_fine=21)
        self.assertEqual(matrices.b0_fine.shape, (21, 8))
        self.assertEqual(matrices.tau_start, 0.0)
        self.assertEqual(matrices.tau_end, 1.0)
        np.testing.assert_allclose(
            np.sum(matrices.b0_fine, axis=1),
            np.ones(21),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_grid_spacing_is_inclusive_without_count_options(self) -> None:
        np.testing.assert_array_equal(
            inclusive_grid(10.0, 20.0, 4.0),
            np.asarray([10.0, 14.0, 18.0]),
        )

    def test_run_id_is_deterministic_and_configuration_sensitive(self) -> None:
        first = {"degree": 5, "n_control_points": 40}
        reordered = {"n_control_points": 40, "degree": 5}
        changed = {"degree": 3, "n_control_points": 40}
        self.assertEqual(config_run_id(first), config_run_id(reordered))
        self.assertNotEqual(config_run_id(first), config_run_id(changed))


if __name__ == "__main__":
    unittest.main()
