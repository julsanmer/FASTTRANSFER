import unittest

import numpy as np

from experiments.reproduce_gondelach_fig2 import (
    basis_integral_matrix,
    basis_matrix,
    parse_basis_group,
    solve_component_coefficients,
)


class GondelachAnalyticBasisTests(unittest.TestCase):
    def test_analytic_derivatives_and_integrals_match_finite_differences(self) -> None:
        terms = parse_basis_group("CPowPow2PSinPCosP6Sin05P4CosN5")
        tau = np.linspace(0.15, 0.85, 9)
        step = 1.0e-6

        values, derivatives = basis_matrix(terms, tau, n_rev=2)
        values_minus, _ = basis_matrix(terms, tau - step, n_rev=2)
        values_plus, _ = basis_matrix(terms, tau + step, n_rev=2)
        finite_difference = (values_plus - values_minus) / (2.0 * step)
        np.testing.assert_allclose(derivatives, finite_difference, rtol=2.0e-8, atol=2.0e-8)

        integrals_minus = basis_integral_matrix(terms, tau - step, n_rev=2)
        integrals_plus = basis_integral_matrix(terms, tau + step, n_rev=2)
        integral_derivative = (integrals_plus - integrals_minus) / (2.0 * step)
        np.testing.assert_allclose(integral_derivative, values, rtol=2.0e-8, atol=2.0e-8)

    def test_component_constraints_use_exact_antiderivatives(self) -> None:
        terms = parse_basis_group("CPowPow2PSinPCos")
        tau = np.linspace(0.0, 1.0, 5)
        duration = 3.2
        initial_value = 0.2
        final_value = -0.3
        displacement = 1.7
        free = np.asarray([0.25, -0.1])

        coefficients = solve_component_coefficients(
            terms,
            tau,
            n_rev=1,
            tf_days=duration,
            y0=initial_value,
            yf=final_value,
            integral_target=displacement,
            free_coefficients=free,
        )
        endpoint_values, _ = basis_matrix(terms, np.asarray([0.0, 1.0]), n_rev=1)
        final_integrals = basis_integral_matrix(terms, np.asarray([1.0]), n_rev=1)[0]

        self.assertAlmostEqual(float(endpoint_values[0] @ coefficients), initial_value, places=13)
        self.assertAlmostEqual(float(endpoint_values[1] @ coefficients), final_value, places=13)
        self.assertAlmostEqual(float(duration * (final_integrals @ coefficients)), displacement, places=13)

if __name__ == "__main__":
    unittest.main()
